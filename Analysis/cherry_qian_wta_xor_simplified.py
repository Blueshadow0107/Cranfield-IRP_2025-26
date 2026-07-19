"""
Simplified Cherry-Qian-style WTA for XOR (well-mixed, scalar readout).

Reactions:
    X1 + W11 -> X1 + H1
    X2 + W12 -> X2 + H1
    X1 + W21 -> X1 + H2
    X2 + W22 -> X2 + H2
    H1 + H2 -> ∅                  (mutual annihilation / WTA)
    H1 + R1 -> H1 + Y             (output production)
    H2 + R2 -> H2 + Y
    H1 -> ∅, H2 -> ∅, Y -> ∅      (degradation)

X1, X2 are pinned input concentrations.
Learned parameters: W11, W12, W21, W22, R1, R2.
Fixed degradation rates are chosen to give reasonable steady-state dynamics.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
T_FINAL = 30.0
INPUTS = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
TARGETS = np.array([0.0, 1.0, 1.0, 0.0])

# Fixed degradation rates
K_DEG_H = 0.1
K_DEG_Y = 0.1
K_ANNIH = 1.0

PARAM_CLIP = 5.0

# -----------------------------------------------------------------------------
def pack_params(W, R):
    """W is (2,2), R is (2,)."""
    return np.concatenate([W.ravel(), R.ravel()])


def unpack_params(p):
    W = p[0:4].reshape((2, 2))
    R = p[4:6]
    return W, R


def clip_params(p):
    W, R = unpack_params(p)
    W = np.clip(W, 0.0, PARAM_CLIP)
    R = np.clip(R, 0.0, PARAM_CLIP)
    return pack_params(W, R)


# -----------------------------------------------------------------------------
def rhs(t, state, x1, x2, W, R):
    H1, H2, Y = state
    W11, W12, W21, W22 = W.ravel()
    R1, R2 = R

    # Production of hidden units
    dH1 = W11 * x1 + W12 * x2 - K_ANNIH * H1 * H2 - K_DEG_H * H1
    dH2 = W21 * x1 + W22 * x2 - K_ANNIH * H1 * H2 - K_DEG_H * H2

    # Output production
    dY = R1 * H1 + R2 * H2 - K_DEG_Y * Y

    return [dH1, dH2, dY]


def simulate(x1, x2, params, t_final=T_FINAL):
    params = clip_params(params)
    W, R = unpack_params(params)
    sol = solve_ivp(
        rhs, [0.0, t_final], [0.0, 0.0, 0.0],
        args=(x1, x2, W, R),
        method='LSODA', rtol=1e-7, atol=1e-9, dense_output=True,
    )
    return sol


def predict(params, t_final=T_FINAL):
    outs = []
    for (x1, x2) in INPUTS:
        sol = simulate(x1, x2, params, t_final)
        if not sol.success or np.any(~np.isfinite(sol.y)):
            return None
        outs.append(max(sol.y[2, -1], 0.0))
    return np.array(outs)


# -----------------------------------------------------------------------------
def loss(params):
    preds = predict(params)
    if preds is None:
        return 1e6
    mse = np.mean((preds - TARGETS) ** 2)
    penalty = 0.0
    if np.any(preds < 0):
        penalty += 1e4
    if np.any(preds > 5.0):
        penalty += 1e4
    reg = 1e-3 * np.sum(params ** 2)
    return mse + penalty + reg


def train_one(seed=42):
    rng = np.random.default_rng(seed)
    p0 = rng.uniform(0.0, 1.0, size=6)  # all positive
    bounds = [(0.0, PARAM_CLIP)] * 6
    result = minimize(
        loss, p0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-7},
    )
    return result.x, result.fun


def train(n_restarts=10):
    best_loss = np.inf
    best_params = None
    for seed in range(n_restarts):
        try:
            p_opt, val = train_one(seed)
        except Exception as e:
            print(f"restart {seed:2d}: failed ({e})")
            continue
        print(f"restart {seed:2d}: loss = {val:.6f}")
        if val < best_loss:
            best_loss = val
            best_params = p_opt
    print(f"\nBest loss over {n_restarts} restarts: {best_loss:.6f}")
    if best_params is not None:
        print("Best predictions:", predict(best_params))
    return best_params


# -----------------------------------------------------------------------------
def plot_results(params, title="Simplified WTA XOR"):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    t_common = np.linspace(0.0, T_FINAL, 500)
    for idx, (x1, x2) in enumerate(INPUTS):
        sol = simulate(x1, x2, params)
        H1, H2, Y = sol.sol(t_common)
        axes[idx].plot(t_common, H1, label='H1', alpha=0.7)
        axes[idx].plot(t_common, H2, label='H2', alpha=0.7)
        axes[idx].plot(t_common, Y, label='Y (output)', linewidth=2)
        axes[idx].axhline(TARGETS[idx], color='r', linestyle='--', label='target')
        axes[idx].set_title(f"X1={x1}, X2={x2}")
        axes[idx].set_xlabel('time')
        axes[idx].set_ylabel('concentration')
        axes[idx].legend()
        axes[idx].grid(True)
    fig.suptitle(title)
    plt.tight_layout()
    fname = 'cherry_qian_wta_xor_simplified_results.png'
    plt.savefig(fname, dpi=150)
    print(f"Saved plot to {fname}")


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    p_opt = train(n_restarts=10)
    if p_opt is not None:
        preds = predict(p_opt)
        print("\nFinal XOR truth table:")
        for (x1, x2), target, pred in zip(INPUTS, TARGETS, preds):
            print(f"  X1={x1}, X2={x2} -> target={target}, predicted={pred:.4f}")
        plot_results(p_opt)
