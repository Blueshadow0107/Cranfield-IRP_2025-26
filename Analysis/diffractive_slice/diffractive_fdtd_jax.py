"""
diffractive_fdtd_jax.py
JAX-based differentiable 2D acoustic FDTD for a diffractive neural layer.

Uses jax.lax.scan for memory-efficient differentiation through long rollouts.
Scatterer presence is binary via Straight-Through Estimator (STE).
Task: 2-input XOR with 4 input channels (2 signal + 2 bias).
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, custom_jvp
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# Straight-Through Estimator (STE) for JAX
# ============================================================================

@custom_jvp
def ste_binary(x):
    """Forward: hard threshold. Backward: identity gradient."""
    return jnp.where(x > 0, 1.0, 0.0)


@ste_binary.defjvp
def _ste_jvp(primals, tangents):
    x, = primals
    g, = tangents
    return ste_binary(x), g


# ============================================================================
# Physical constants
# ============================================================================

DX = 5e-5              # 50 µm
C = 1500.0             # m/s (water)
FREQ = 1.0e6           # 1 MHz
LAMBDA = C / FREQ      # 1.5 mm
CFL = 0.5
DT = CFL * DX / C
COEF = (C * DT / DX) ** 2
STEPS_PER_PERIOD = int(round((1.0 / FREQ) / DT))

# Domain
NX, NY = 300, 200


# ============================================================================
# Fixed geometry definitions (JAX arrays, computed once at import)
# ============================================================================

# --- Inputs: 4 waveguides on left ---
N_INPUTS = 4
INPUT_CENTRES_Y = jnp.array([45, 75, 125, 155])
INPUT_WIDTH = 9
INPUT_LENGTH = 40

# --- Scatterer region: trainable binary super-pixel grid ---
SCAT_NX, SCAT_NY = 10, 10          # 10x10 = 100 binary params
SUPERPIXEL = 5                     # each param controls a 5x5 cell block = 250 µm
SCAT_X0 = 125
SCAT_Y0 = 75

# --- Outputs: 2 detectors on right ---
DETECTOR_X = 280
DETECTOR_Y = jnp.array([85, 115])
N_DETECTORS = len(DETECTOR_Y)

# Precompute fixed masks
_fixed_bc = jnp.zeros((NX, NY))
for cy in INPUT_CENTRES_Y:
    y_min = int(round(float(cy) - INPUT_WIDTH // 2))
    y_max = int(round(float(cy) + INPUT_WIDTH // 2))
    if y_min > 0:
        _fixed_bc = _fixed_bc.at[0:INPUT_LENGTH, y_min].set(1.0)
    if y_max < NY - 1:
        _fixed_bc = _fixed_bc.at[0:INPUT_LENGTH, y_max].set(1.0)

# Source masks (boolean as float for JAX indexing)
_source_masks = []
_src_x = 2
for cy in INPUT_CENTRES_Y:
    m = jnp.zeros((NX, NY))
    m = m.at[_src_x, int(round(float(cy)))].set(1.0)
    _source_masks.append(m)

_source_mask_any = jnp.zeros((NX, NY))
for m in _source_masks:
    _source_mask_any = _source_mask_any + m

# Detector indices as Python ints for indexing inside traced functions
_DETECTOR_Y_INT = [int(dy) for dy in DETECTOR_Y]

# Detector masks
_detector_masks = []
for dy in _DETECTOR_Y_INT:
    m = jnp.zeros((NX, NY))
    m = m.at[DETECTOR_X, dy].set(1.0)
    _detector_masks.append(m)


# ============================================================================
# Build BC mask from trainable scatterer parameters
# ============================================================================

def build_bc_mask(theta):
    """theta: flat array of shape (SCAT_NX*SCAT_NY,)."""
    presence = ste_binary(theta).reshape(SCAT_NX, SCAT_NY)
    # Upsample each param to a SUPERPIXEL x SUPERPIXEL block
    presence_up = jnp.repeat(jnp.repeat(presence, SUPERPIXEL, axis=0), SUPERPIXEL, axis=1)
    pad_x = (SCAT_X0, NX - SCAT_X0 - SCAT_NX * SUPERPIXEL)
    pad_y = (SCAT_Y0, NY - SCAT_Y0 - SCAT_NY * SUPERPIXEL)
    scatterer = jnp.pad(presence_up, (pad_x, pad_y), mode='constant')
    bc = _fixed_bc + scatterer
    # Clamp to [0,1] in case of overlap
    bc = jnp.clip(bc, 0.0, 1.0)
    return bc


# ============================================================================
# Laplacian with ghost-cell Neumann reflection
# ============================================================================

def laplacian(p, bc_mask):
    """
    5-point stencil with ghost-cell reflection at walls.
    Fully differentiable via smooth blending.
    """
    # Shifted neighbors with zero padding
    left_s = jnp.pad(p[:-1, :], ((1, 0), (0, 0)), mode='constant')
    right_s = jnp.pad(p[1:, :], ((0, 1), (0, 0)), mode='constant')
    bottom_s = jnp.pad(p[:, :-1], ((0, 0), (1, 0)), mode='constant')
    top_s = jnp.pad(p[:, 1:], ((0, 0), (0, 1)), mode='constant')

    # Wall indicators shifted to align
    wall_left = jnp.pad(bc_mask[:-1, :], ((1, 0), (0, 0)), mode='constant')
    wall_right = jnp.pad(bc_mask[1:, :], ((0, 1), (0, 0)), mode='constant')
    wall_bottom = jnp.pad(bc_mask[:, :-1], ((0, 0), (1, 0)), mode='constant')
    wall_top = jnp.pad(bc_mask[:, 1:], ((0, 0), (0, 1)), mode='constant')

    # Ghost-cell reflection
    left = wall_left * p + (1.0 - wall_left) * left_s
    right = wall_right * p + (1.0 - wall_right) * right_s
    bottom = wall_bottom * p + (1.0 - wall_bottom) * bottom_s
    top = wall_top * p + (1.0 - wall_top) * top_s

    return left + right + bottom + top - 4.0 * p


# ============================================================================
# Single FDTD step (pure function for scan)
# ============================================================================

def make_fdtd_step(bc_mask, source_phases, src_amp=1.0):
    """Returns a step function compatible with jax.lax.scan."""
    omega = 2.0 * jnp.pi * FREQ

    def step(carry, n):
        p_prev, p_curr = carry
        t = n * DT

        # Hard source injection
        src_vals = jnp.zeros((NX, NY))
        for i, phase in enumerate(source_phases):
            val = src_amp * jnp.sin(omega * t + phase)
            src_vals = src_vals + _source_masks[i] * val
        p_curr = jnp.where(_source_mask_any > 0.5, src_vals, p_curr)

        # Leapfrog update
        lap = laplacian(p_curr, bc_mask)
        p_next = 2.0 * p_curr - p_prev + COEF * lap

        # Freeze walls
        p_next = bc_mask * p_curr + (1.0 - bc_mask) * p_next

        # Record detectors
        detector_readings = jnp.stack([
            p_curr[DETECTOR_X, _DETECTOR_Y_INT[d]]
            for d in range(N_DETECTORS)
        ])

        return (p_curr, p_next), detector_readings

    return step


def run_fdtd(bc_mask, source_phases, nt=1200, src_amp=1.0):
    """
    Run FDTD using jax.lax.scan for memory-efficient differentiation.
    Returns detector traces array (nt, N_DETECTORS) and final pressure field.
    """
    # Initialise fields
    p_prev = jnp.zeros((NX, NY))
    p_curr = jnp.zeros((NX, NY))

    step_fn = make_fdtd_step(bc_mask, source_phases, src_amp)

    # scan: carry = (p_prev, p_curr), ys = detector readings
    (p_final, _), probe_traces = jax.lax.scan(step_fn, (p_prev, p_curr), jnp.arange(nt))

    return probe_traces, p_final


# ============================================================================
# Loss function for XOR task
# ============================================================================

# 4 input patterns for 2-bit XOR (channels 2,3 are bias)
XOR_PATTERNS = jnp.array([
    [0.0,       0.0,       jnp.pi/2,  jnp.pi/2],   # (0,0) -> class 0
    [0.0,       jnp.pi,    jnp.pi/2,  jnp.pi/2],   # (0,1) -> class 1
    [jnp.pi,    0.0,       jnp.pi/2,  jnp.pi/2],   # (1,0) -> class 1
    [jnp.pi,    jnp.pi,    jnp.pi/2,  jnp.pi/2],   # (1,1) -> class 0
])

XOR_TARGETS = jnp.array([
    [1.0, 0.0],   # class 0
    [0.0, 1.0],   # class 1
    [0.0, 1.0],   # class 1
    [1.0, 0.0],   # class 0
])


def softmax(logits):
    """Numerically stable softmax."""
    max_logit = jnp.max(logits)
    exps = jnp.exp(logits - max_logit)
    return exps / jnp.sum(exps)


def cross_entropy(probs, target):
    """Binary cross-entropy."""
    eps = 1e-12
    probs = jnp.clip(probs, eps, 1.0 - eps)
    return -jnp.sum(target * jnp.log(probs))


def loss_fn(theta, nt=1200, n_periods_measure=3):
    """
    Total loss over all XOR patterns.
    """
    bc_mask = build_bc_mask(theta)
    total_loss = 0.0

    for pattern, target in zip(XOR_PATTERNS, XOR_TARGETS):
        probe_traces, _ = run_fdtd(bc_mask, pattern, nt=nt)

        # RMS over last N periods for each detector
        window_len = n_periods_measure * STEPS_PER_PERIOD
        rms_values = []
        for d in range(N_DETECTORS):
            window = probe_traces[-window_len:, d]
            rms = jnp.sqrt(jnp.mean(window ** 2) + 1e-12)
            rms_values.append(rms)

        probs = softmax(jnp.array(rms_values))
        total_loss = total_loss + cross_entropy(probs, target)

    return total_loss


# ============================================================================
# JIT-compiled gradient evaluation
# ============================================================================

loss_and_grad = jit(lambda theta, nt: (loss_fn(theta, nt), grad(loss_fn)(theta, nt)))


# ============================================================================
# Training
# ============================================================================

def train(theta0=None, n_iter=500, lr=0.5, nt=1200, verbose=True):
    """
    Vanilla gradient descent with momentum.
    Returns optimised theta and loss history.
    """
    if theta0 is None:
        theta = jnp.zeros(SCAT_NX * SCAT_NY)
    else:
        theta = jnp.array(theta0)

    # Simple momentum
    v = jnp.zeros_like(theta)
    beta = 0.9

    losses = []
    for it in range(n_iter):
        g = grad(loss_fn)(theta, nt=nt)
        v = beta * v - lr * g
        theta = theta + v
        loss_val = float(loss_fn(theta, nt=nt))
        losses.append(loss_val)

        if verbose and it % 20 == 0:
            print(f"Iter {it:4d}/{n_iter}  loss = {loss_val:.4f}")

    return np.array(theta), losses


# ============================================================================
# Evaluation / visualisation
# ============================================================================

def evaluate(theta, nt=1200):
    """Run all XOR patterns and print classification results."""
    bc_mask = build_bc_mask(theta)
    print("\n=== XOR Evaluation ===")
    for idx, (pattern, target) in enumerate(zip(XOR_PATTERNS, XOR_TARGETS)):
        probe_traces, final_field = run_fdtd(bc_mask, pattern, nt=nt)
        window_len = 3 * STEPS_PER_PERIOD
        rms = []
        for d in range(N_DETECTORS):
            w = probe_traces[-window_len:, d]
            rms.append(float(jnp.sqrt(jnp.mean(w ** 2))))
        probs = softmax(jnp.array(rms))
        pred = int(jnp.argmax(probs))
        true = int(jnp.argmax(target))
        print(f"Pattern {idx}: phases=[{float(pattern[0]):.2f},{float(pattern[1]):.2f}]  "
              f"RMS=[{rms[0]:.4f},{rms[1]:.4f}]  "
              f"probs=[{float(probs[0]):.3f},{float(probs[1]):.3f}]  "
              f"pred={pred}  true={true}  {'OK' if pred==true else 'FAIL'}")
    return bc_mask


def plot_results(theta, losses, nt=1200):
    """Plot loss curve, scatterer config, and field snapshots."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Loss curve
    ax = axes[0, 0]
    ax.plot(losses)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.grid(True)

    # Scatterer configuration
    bc_mask = build_bc_mask(theta)
    ax = axes[0, 1]
    im = ax.imshow(np.array(bc_mask).T, origin='lower', cmap='gray_r',
                   extent=[0, NX*DX*1e3, 0, NY*DX*1e3])
    ax.set_title("Scatterer Configuration")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Field snapshots for each XOR pattern
    for idx in range(4):
        if idx < 2:
            ax = axes[1, idx]
        else:
            ax = axes[0, 2] if idx == 2 else axes[1, idx - 2]
        pattern = XOR_PATTERNS[idx]
        _, field = run_fdtd(bc_mask, pattern, nt=nt)
        vmax = float(jnp.max(jnp.abs(field)))
        im = ax.imshow(np.array(field).T, origin='lower', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax,
                       extent=[0, NX*DX*1e3, 0, NY*DX*1e3])
        ax.set_title(f"Pattern {idx}: ({int(jnp.argmax(XOR_TARGETS[idx]))})")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig("diffractive_results_jax.png", dpi=150)
    plt.show()
    print("Saved diffractive_results_jax.png")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Diffractive Acoustic Neural Network — XOR Task (JAX)")
    print("=" * 60)
    print(f"Domain: {NX} x {NY} cells = {NX*DX*1e3:.1f} x {NY*DX*1e3:.1f} mm")
    print(f"Wavelength: {LAMBDA*1e3:.2f} mm = {LAMBDA/DX:.0f} cells")
    print(f"Steps/period: {STEPS_PER_PERIOD}")
    print(f"Scatterer grid: {SCAT_NX} x {SCAT_NY} = {SCAT_NX*SCAT_NY} params")
    print()

    # Initial random scatterers
    np.random.seed(42)
    theta0 = np.random.randn(SCAT_NX * SCAT_NY) * 0.1

    print("Training...")
    theta_opt, losses = train(theta0, n_iter=300, lr=0.3, nt=1000)

    print("\nEvaluating...")
    evaluate(theta_opt, nt=1000)

    print("\nPlotting...")
    plot_results(theta_opt, losses, nt=1000)
