"""
diffractive_fdtd.py
Self-contained differentiable 2D acoustic FDTD for a diffractive neural layer.

Uses autograd for automatic differentiation through the time-stepping loop.
Scatterer presence is binary (0/1) via Straight-Through Estimator (STE).
Task: 2-input XOR with 4 input channels (2 signal + 2 bias).
"""

import autograd.numpy as np
from autograd import grad
from autograd.extend import primitive, defvjp
import matplotlib.pyplot as plt

# ============================================================================
# Straight-Through Estimator (STE)
# ============================================================================

@primitive
def ste_binary(x):
    """Forward: hard threshold. Backward: identity gradient."""
    return np.where(x > 0, 1.0, 0.0)


def _ste_vjp(ans, x):
    return lambda g: g


defvjp(ste_binary, _ste_vjp)


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
# Fixed geometry definitions
# ============================================================================

# --- Inputs: 4 waveguides on left ---
N_INPUTS = 4
INPUT_CENTRES_Y = np.array([45, 75, 125, 155])
INPUT_WIDTH = 9
INPUT_LENGTH = 40

# --- Scatterer region: trainable binary grid ---
SCAT_NX, SCAT_NY = 10, 10          # 10x10 = 100 binary params
SCAT_X0 = 130
SCAT_Y0 = 70

# --- Outputs: 2 detectors on right ---
DETECTOR_X = 280
DETECTOR_Y = np.array([85, 115])
N_DETECTORS = len(DETECTOR_Y)

# Precompute fixed masks
_fixed_bc = np.zeros((NX, NY))
for cy in INPUT_CENTRES_Y:
    y_min = int(round(cy - INPUT_WIDTH // 2))
    y_max = int(round(cy + INPUT_WIDTH // 2))
    if y_min > 0:
        _fixed_bc[0:INPUT_LENGTH, y_min] = 1
    if y_max < NY - 1:
        _fixed_bc[0:INPUT_LENGTH, y_max] = 1

# Source masks (boolean)
_source_masks = []
_src_x = 2
for cy in INPUT_CENTRES_Y:
    m = np.zeros((NX, NY), dtype=bool)
    m[_src_x, int(round(cy))] = True
    _source_masks.append(m)

_source_mask_any = np.zeros((NX, NY), dtype=bool)
for m in _source_masks:
    _source_mask_any |= m

# Detector masks (boolean)
_detector_masks = []
for dy in DETECTOR_Y:
    m = np.zeros((NX, NY), dtype=bool)
    m[DETECTOR_X, dy] = True
    _detector_masks.append(m)


# ============================================================================
# Build BC mask from trainable scatterer parameters
# ============================================================================

def build_bc_mask(theta):
    """theta: flat array of shape (SCAT_NX*SCAT_NY,)."""
    presence = ste_binary(theta).reshape(SCAT_NX, SCAT_NY)
    pad_x = (SCAT_X0, NX - SCAT_X0 - SCAT_NX)
    pad_y = (SCAT_Y0, NY - SCAT_Y0 - SCAT_NY)
    scatterer = np.pad(presence, (pad_x, pad_y), mode='constant')
    bc = _fixed_bc + scatterer
    return bc


# ============================================================================
# Laplacian with ghost-cell Neumann reflection (fully differentiable)
# ============================================================================

def laplacian(p, bc_mask):
    """
    5-point stencil with ghost-cell reflection at walls.
    Uses smooth blending (no boolean ops) so gradients flow through bc_mask.
    """
    # Shifted neighbor arrays (pad with zeros at boundaries)
    left_s = np.pad(p[:-1, :], ((1, 0), (0, 0)), mode='constant')
    right_s = np.pad(p[1:, :], ((0, 1), (0, 0)), mode='constant')
    bottom_s = np.pad(p[:, :-1], ((0, 0), (1, 0)), mode='constant')
    top_s = np.pad(p[:, 1:], ((0, 0), (0, 1)), mode='constant')

    # Wall indicators shifted to align with neighbor direction (float, 0 or 1)
    wall_left = np.pad(bc_mask[:-1, :], ((1, 0), (0, 0)), mode='constant')
    wall_right = np.pad(bc_mask[1:, :], ((0, 1), (0, 0)), mode='constant')
    wall_bottom = np.pad(bc_mask[:, :-1], ((0, 0), (1, 0)), mode='constant')
    wall_top = np.pad(bc_mask[:, 1:], ((0, 0), (0, 1)), mode='constant')

    # Ghost-cell reflection: if wall, reflect (use p itself); else use neighbor
    left = wall_left * p + (1.0 - wall_left) * left_s
    right = wall_right * p + (1.0 - wall_right) * right_s
    bottom = wall_bottom * p + (1.0 - wall_bottom) * bottom_s
    top = wall_top * p + (1.0 - wall_top) * top_s

    return left + right + bottom + top - 4.0 * p


# ============================================================================
# Single FDTD run for one input pattern
# ============================================================================

def run_fdtd(bc_mask, source_phases, nt=1200, src_amp=1.0):
    """
    Run FDTD. Returns list of detector traces and final pressure field.
    All operations are autograd-safe (rebinding, no mutation).
    """
    # Initialise fields from bc_mask so they are traced arrays
    p_prev = bc_mask * 0.0
    p_curr = bc_mask * 0.0
    p_next = bc_mask * 0.0

    probe_traces = [[] for _ in range(N_DETECTORS)]
    omega = 2.0 * np.pi * FREQ

    for n in range(nt):
        t = n * DT

        # Hard source injection (autograd-safe)
        src_vals = np.zeros((NX, NY))
        for i, phase in enumerate(source_phases):
            val = src_amp * np.sin(omega * t + phase)
            src_vals = np.where(_source_masks[i], val, src_vals)
        p_curr = np.where(_source_mask_any, src_vals, p_curr)

        # Leapfrog
        lap = laplacian(p_curr, bc_mask)
        p_next = 2.0 * p_curr - p_prev + COEF * lap

        # Freeze walls (smooth blending, differentiable)
        p_next = bc_mask * p_curr + (1.0 - bc_mask) * p_next

        # Swap (rebinding only)
        p_prev, p_curr, p_next = p_curr, p_next, p_prev

        # Record detectors
        for d in range(N_DETECTORS):
            probe_traces[d].append(p_curr[DETECTOR_X, DETECTOR_Y[d]])

    return probe_traces, p_curr


# ============================================================================
# Loss function for XOR task
# ============================================================================

# 4 input patterns for 2-bit XOR (channels 2,3 are bias)
XOR_PATTERNS = np.array([
    [0.0,       0.0,       np.pi/2,  np.pi/2],   # (0,0) -> class 0
    [0.0,       np.pi,     np.pi/2,  np.pi/2],   # (0,1) -> class 1
    [np.pi,     0.0,       np.pi/2,  np.pi/2],   # (1,0) -> class 1
    [np.pi,     np.pi,     np.pi/2,  np.pi/2],   # (1,1) -> class 0
])

XOR_TARGETS = np.array([
    [1.0, 0.0],   # class 0
    [0.0, 1.0],   # class 1
    [0.0, 1.0],   # class 1
    [1.0, 0.0],   # class 0
])


def softmax(logits):
    """Numerically stable softmax."""
    logits = np.array(logits)
    max_logit = np.max(logits)
    exps = np.exp(logits - max_logit)
    return exps / np.sum(exps)


def cross_entropy(probs, target):
    """Binary cross-entropy."""
    eps = 1e-12
    probs = np.clip(probs, eps, 1.0 - eps)
    return -np.sum(target * np.log(probs))


def loss_fn(theta, nt=1200, n_periods_measure=3):
    """
    Total loss over all XOR patterns.

    For each pattern, run FDTD, measure RMS at both detectors over the
    last `n_periods_measure` periods, softmax -> cross-entropy.
    """
    bc_mask = build_bc_mask(theta)
    total_loss = 0.0

    for pattern, target in zip(XOR_PATTERNS, XOR_TARGETS):
        probe_traces, _ = run_fdtd(bc_mask, pattern, nt=nt)

        # RMS over last N periods for each detector
        window_len = n_periods_measure * STEPS_PER_PERIOD
        rms_values = []
        for d in range(N_DETECTORS):
            window = np.array(probe_traces[d][-window_len:])
            rms = np.sqrt(np.mean(window ** 2) + 1e-12)
            rms_values.append(rms)

        probs = softmax(np.array(rms_values))
        total_loss += cross_entropy(probs, target)

    return total_loss


# ============================================================================
# Training
# ============================================================================

def train(theta0=None, n_iter=500, lr=0.5, nt=1200, verbose=True):
    """
    Vanilla gradient descent with momentum.
    Returns optimised theta and loss history.
    """
    if theta0 is None:
        theta = np.zeros(SCAT_NX * SCAT_NY)
    else:
        theta = theta0.copy()

    grad_fn = grad(loss_fn)

    # Simple momentum
    v = np.zeros_like(theta)
    beta = 0.9

    losses = []
    for it in range(n_iter):
        g = grad_fn(theta, nt=nt)
        v = beta * v - lr * g
        theta = theta + v
        losses.append(float(loss_fn(theta, nt=nt)))

        if verbose and it % 20 == 0:
            print(f"Iter {it:4d}/{n_iter}  loss = {losses[-1]:.4f}")

    return theta, losses


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
            w = np.array(probe_traces[d][-window_len:])
            rms.append(np.sqrt(np.mean(w ** 2)))
        probs = softmax(np.array(rms))
        pred = np.argmax(probs)
        true = np.argmax(target)
        print(f"Pattern {idx}: phases=[{pattern[0]:.2f},{pattern[1]:.2f}]  "
              f"RMS=[{rms[0]:.4f},{rms[1]:.4f}]  "
              f"probs=[{probs[0]:.3f},{probs[1]:.3f}]  "
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
    im = ax.imshow(bc_mask.T, origin='lower', cmap='gray_r',
                   extent=[0, NX*DX*1e3, 0, NY*DX*1e3])
    ax.set_title("Scatterer Configuration")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Field snapshots for each XOR pattern
    for idx in range(4):
        ax = axes[1, idx] if idx < 2 else axes[0, 2]
        if idx >= 2:
            ax = axes[1, idx - 2]
        pattern = XOR_PATTERNS[idx]
        _, field = run_fdtd(bc_mask, pattern, nt=nt)
        vmax = np.abs(field).max()
        im = ax.imshow(field.T, origin='lower', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax,
                       extent=[0, NX*DX*1e3, 0, NY*DX*1e3])
        ax.set_title(f"Pattern {idx}: ({np.argmax(XOR_TARGETS[idx])})")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig("diffractive_results.png", dpi=150)
    plt.show()
    print("Saved diffractive_results.png")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Diffractive Acoustic Neural Network — XOR Task")
    print("=" * 60)
    print(f"Domain: {NX} x {NY} cells = {NX*DX*1e3:.1f} x {NY*DX*1e3:.1f} mm")
    print(f"Wavelength: {LAMBDA*1e3:.2f} mm = {LAMBDA/DX:.0f} cells")
    print(f"Steps/period: {STEPS_PER_PERIOD}")
    print(f"Scatterer grid: {SCAT_NX} x {SCAT_NY} = {SCAT_NX*SCAT_NY} params")
    print()

    # Initial random scatterers (small random values)
    np.random.seed(42)
    theta0 = np.random.randn(SCAT_NX * SCAT_NY) * 0.1

    print("Training...")
    theta_opt, losses = train(theta0, n_iter=300, lr=0.3, nt=1000)

    print("\nEvaluating...")
    evaluate(theta_opt, nt=1000)

    print("\nPlotting...")
    plot_results(theta_opt, losses, nt=1000)
