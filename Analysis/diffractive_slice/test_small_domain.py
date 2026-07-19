"""
test_small_domain.py
Minimal validation of differentiable FDTD on a tiny domain.
Validates: forward pass, backward pass, loss decrease.
"""

import autograd.numpy as np
from autograd import grad
from autograd.extend import primitive, defvjp

# ============================================================================
# STE
# ============================================================================
@primitive
def ste_binary(x):
    return np.where(x > 0, 1.0, 0.0)

def _ste_vjp(ans, x):
    return lambda g: g

defvjp(ste_binary, _ste_vjp)

# ============================================================================
# Physical constants
# ============================================================================
DX = 5e-5
C = 1500.0
FREQ = 1.0e6
CFL = 0.5
DT = CFL * DX / C
COEF = (C * DT / DX) ** 2
STEPS_PER_PERIOD = int(round((1.0 / FREQ) / DT))

# ============================================================================
# Small domain
# ============================================================================
NX, NY = 80, 60

# 2 inputs, close together on left
N_INPUTS = 2
INPUT_X = 2
INPUT_Y = np.array([20, 40])

# 4x4 scatterer grid, close to inputs
SCAT_NX, SCAT_NY = 4, 4
SCAT_X0, SCAT_Y0 = 20, 25

# 2 detectors on right
DET_X = 75
DET_Y = np.array([20, 40])

# Source masks
_source_masks = []
for y in INPUT_Y:
    m = np.zeros((NX, NY), dtype=bool)
    m[INPUT_X, y] = True
    _source_masks.append(m)
_source_any = np.zeros((NX, NY), dtype=bool)
for m in _source_masks:
    _source_any |= m

# ============================================================================
# Build BC mask
# ============================================================================
def build_bc_mask(theta):
    presence = ste_binary(theta).reshape(SCAT_NX, SCAT_NY)
    scatterer = np.pad(presence, ((SCAT_X0, NX-SCAT_X0-SCAT_NX), (SCAT_Y0, NY-SCAT_Y0-SCAT_NY)), mode='constant')
    return scatterer  # no fixed walls in this tiny test

# ============================================================================
# Laplacian (smooth, differentiable)
# ============================================================================
def laplacian(p, bc_mask):
    left_s = np.pad(p[:-1, :], ((1, 0), (0, 0)), mode='constant')
    right_s = np.pad(p[1:, :], ((0, 1), (0, 0)), mode='constant')
    bottom_s = np.pad(p[:, :-1], ((0, 0), (1, 0)), mode='constant')
    top_s = np.pad(p[:, 1:], ((0, 0), (0, 1)), mode='constant')

    wall_left = np.pad(bc_mask[:-1, :], ((1, 0), (0, 0)), mode='constant')
    wall_right = np.pad(bc_mask[1:, :], ((0, 1), (0, 0)), mode='constant')
    wall_bottom = np.pad(bc_mask[:, :-1], ((0, 0), (1, 0)), mode='constant')
    wall_top = np.pad(bc_mask[:, 1:], ((0, 0), (0, 1)), mode='constant')

    left = wall_left * p + (1.0 - wall_left) * left_s
    right = wall_right * p + (1.0 - wall_right) * right_s
    bottom = wall_bottom * p + (1.0 - wall_bottom) * bottom_s
    top = wall_top * p + (1.0 - wall_top) * top_s

    return left + right + bottom + top - 4.0 * p

# ============================================================================
# FDTD
# ============================================================================
def run_fdtd(bc_mask, source_phases, nt=100, src_amp=1.0):
    p_prev = bc_mask * 0.0
    p_curr = bc_mask * 0.0
    p_next = bc_mask * 0.0

    probe_traces = [[] for _ in range(len(DET_Y))]
    omega = 2.0 * np.pi * FREQ

    for n in range(nt):
        t = n * DT

        # Sources
        src_vals = np.zeros((NX, NY))
        for i, phase in enumerate(source_phases):
            val = src_amp * np.sin(omega * t + phase)
            src_vals = np.where(_source_masks[i], val, src_vals)
        p_curr = np.where(_source_any, src_vals, p_curr)

        # Leapfrog
        lap = laplacian(p_curr, bc_mask)
        p_next = 2.0 * p_curr - p_prev + COEF * lap
        p_next = bc_mask * p_curr + (1.0 - bc_mask) * p_next
        p_prev, p_curr, p_next = p_curr, p_next, p_prev

        for d in range(len(DET_Y)):
            probe_traces[d].append(p_curr[DET_X, DET_Y[d]])

    return probe_traces, p_curr

# ============================================================================
# XOR task (2 inputs)
# ============================================================================
XOR_PATTERNS = np.array([
    [0.0,       0.0],       # (0,0) -> class 0
    [0.0,       np.pi],     # (0,1) -> class 1
    [np.pi,     0.0],       # (1,0) -> class 1
    [np.pi,     np.pi],     # (1,1) -> class 0
])

XOR_TARGETS = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [0.0, 1.0],
    [1.0, 0.0],
])

def softmax(logits):
    logits = np.array(logits)
    max_logit = np.max(logits)
    exps = np.exp(logits - max_logit)
    return exps / np.sum(exps)

def cross_entropy(probs, target):
    eps = 1e-12
    probs = np.clip(probs, eps, 1.0 - eps)
    return -np.sum(target * np.log(probs))

def loss_fn(theta, nt=100):
    bc_mask = build_bc_mask(theta)
    total_loss = 0.0
    for pattern, target in zip(XOR_PATTERNS, XOR_TARGETS):
        probes, _ = run_fdtd(bc_mask, pattern, nt=nt)
        window_len = 3 * STEPS_PER_PERIOD
        rms = []
        for d in range(len(DET_Y)):
            w = np.array(probes[d][-window_len:])
            rms.append(np.sqrt(np.mean(w ** 2) + 1e-12))
        probs = softmax(np.array(rms))
        total_loss += cross_entropy(probs, target)
    return total_loss

# ============================================================================
# Quick validation
# ============================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("Small-domain validation")
    print(f"Domain: {NX}x{NY}, Scatterers: {SCAT_NX}x{SCAT_NY}={SCAT_NX*SCAT_NY}")
    print(f"Steps/period: {STEPS_PER_PERIOD}")
    print()

    np.random.seed(0)
    theta = np.random.randn(SCAT_NX * SCAT_NY) * 0.1

    print("1. Forward pass...")
    loss0 = loss_fn(theta, nt=100)
    print(f"   Loss = {loss0:.4f}")

    print("2. Gradient...")
    g = grad(loss_fn)(theta, nt=100)
    print(f"   |grad| = {np.linalg.norm(g):.4f}")
    print(f"   finite = {np.all(np.isfinite(g))}")
    print(f"   nonzero = {np.sum(np.abs(g) > 1e-10)}/{len(g)}")

    print("3. A few training steps...")
    lr = 0.5
    for step in range(10):
        g = grad(loss_fn)(theta, nt=100)
        theta = theta - lr * g
        loss = loss_fn(theta, nt=100)
        print(f"   Step {step+1}: loss = {loss:.4f}")

    print("\nDone.")
