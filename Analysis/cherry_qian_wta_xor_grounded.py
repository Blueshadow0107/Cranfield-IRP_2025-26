"""
Full Cherry-Qian WTA for XOR (well-mixed, scalar readout), physically grounded.

Based on Cherry & Qian, Nature 2018.

Reactions (all mass-action, all rates positive):
    X_i + W_ij + XF_i -> X_i + P_ij        (weight multiplication; W_ij consumed)
    P_ij + SG_j        -> S_j               (summation; SG_j consumed)
    S_1 + S_2 + Anh   -> ∅                  (pairwise annihilation)
    S_1 + RG_1 + YF_1 -> S_1 + Y_1         (signal restoration; RG_1 consumed)
    S_2 + RG_2 + YF_2 -> S_2 + Y_2
    P_ij, S_j, Y_j    -> ∅                  (degradation / dilution)

X1, X2 are pinned input concentrations.
XF_i, SG_j, Anh, YF_j are fixed auxiliary/fuel species in excess.
Learned parameters: initial concentrations of W_ij and RG_j.
Output read from Y_1 (target [0,1,1,0]).
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
T_FINAL = 30.0
INPUTS = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
TARGETS = np.array([0.0, 1.0, 1.0, 0.0])

# Fixed auxiliary species concentrations (fuels / gates / annihilator)
XF1 = 10.0
XF2 = 10.0
SG1 = 5.0
SG2 = 5.0
ANH = 5.0
YF1 = 10.0
YF2 = 10.0

# Fixed rate constants
K_WEIGHT = 1.0       # X_i + W_ij + XF_i -> X_i + P_ij
K_SUM = 1.0          # P_ij + SG_j -> S_j
K_ANNIH = 100.0      # S_1 + S_2 + Anh -> ∅
K_RESTORE = 1.0      # S_j + RG_j + YF_j -> S_j + Y_j
K_DEG_P = 0.001      # P_ij degradation
K_DEG_S = 0.001      # S_j degradation
K_DEG_Y = 0.01       # Y_j degradation

PARAM_CLIP = 5.0

# State index mapping
IDX_W11, IDX_W12, IDX_W21, IDX_W22 = 0, 1, 2, 3
IDX_P11, IDX_P12, IDX_P21, IDX_P22 = 4, 5, 6, 7
IDX_S1, IDX_S2 = 8, 9
IDX_RG1, IDX_RG2 = 10, 11
IDX_Y1, IDX_Y2 = 12, 13
N_STATE = 14

# -----------------------------------------------------------------------------
def pack_params(W0, RG0):
    """W0 is (2,2), RG0 is (2,)."""
    return np.concatenate([W0.ravel(), RG0.ravel()])


def unpack_params(p):
    W0 = p[0:4].reshape((2, 2))
    RG0 = p[4:6]
    return W0, RG0


def clip_params(p):
    W0, RG0 = unpack_params(p)
    W0 = np.clip(W0, 0.0, PARAM_CLIP)
    RG0 = np.clip(RG0, 0.0, PARAM_CLIP)
    return pack_params(W0, RG0)


# -----------------------------------------------------------------------------
def rhs(t, state, x1, x2):
    W11 = state[IDX_W11]
    W12 = state[IDX_W12]
    W21 = state[IDX_W21]
    W22 = state[IDX_W22]

    P11 = state[IDX_P11]
    P12 = state[IDX_P12]
    P21 = state[IDX_P21]
    P22 = state[IDX_P22]

    S1 = state[IDX_S1]
    S2 = state[IDX_S2]

    RG1 = state[IDX_RG1]
    RG2 = state[IDX_RG2]

    Y1 = state[IDX_Y1]
    Y2 = state[IDX_Y2]

    # Weight multiplication
    rW11 = K_WEIGHT * x1 * W11 * XF1
    rW12 = K_WEIGHT * x2 * W12 * XF2
    rW21 = K_WEIGHT * x1 * W21 * XF1
    rW22 = K_WEIGHT * x2 * W22 * XF2

    # Summation
    rP11 = K_SUM * P11 * SG1
    rP12 = K_SUM * P12 * SG1
    rP21 = K_SUM * P21 * SG2
    rP22 = K_SUM * P22 * SG2

    # Annihilation
    rAnnih = K_ANNIH * S1 * S2 * ANH

    # Restoration
    rRestore1 = K_RESTORE * S1 * RG1 * YF1
    rRestore2 = K_RESTORE * S2 * RG2 * YF2

    dstate = np.zeros(N_STATE)

    # Weights are consumed
    dstate[IDX_W11] = -rW11
    dstate[IDX_W12] = -rW12
    dstate[IDX_W21] = -rW21
    dstate[IDX_W22] = -rW22

    # Intermediates produced from weights, consumed by summation
    dstate[IDX_P11] = rW11 - rP11 - K_DEG_P * P11
    dstate[IDX_P12] = rW12 - rP12 - K_DEG_P * P12
    dstate[IDX_P21] = rW21 - rP21 - K_DEG_P * P21
    dstate[IDX_P22] = rW22 - rP22 - K_DEG_P * P22

    # Weighted sums produced from intermediates, annihilated
    dstate[IDX_S1] = rP11 + rP12 - rAnnih - K_DEG_S * S1
    dstate[IDX_S2] = rP21 + rP22 - rAnnih - K_DEG_S * S2

    # Restoration gates consumed
    dstate[IDX_RG1] = -rRestore1
    dstate[IDX_RG2] = -rRestore2

    # Outputs produced from restoration
    dstate[IDX_Y1] = rRestore1 - K_DEG_Y * Y1
    dstate[IDX_Y2] = rRestore2 - K_DEG_Y * Y2

    return dstate


def simulate(x1, x2, params, t_final=T_FINAL):
    params = clip_params(params)
    W0, RG0 = unpack_params(params)
    y0 = np.zeros(N_STATE)
    y0[IDX_W11] = W0[0, 0]
    y0[IDX_W12] = W0[0, 1]
    y0[IDX_W21] = W0[1, 0]
    y0[IDX_W22] = W0[1, 1]
    y0[IDX_RG1] = RG0[0]
    y0[IDX_RG2] = RG0[1]

    sol = solve_ivp(
        rhs, [0.0, t_final], y0,
        args=(x1, x2),
        method='LSODA', rtol=1e-7, atol=1e-9, dense_output=True,
    )
    return sol


def predict(params, t_final=T_FINAL):
    outs = []
    for (x1, x2) in INPUTS:
        sol = simulate(x1, x2, params, t_final)
        if not sol.success or np.any(~np.isfinite(sol.y)):
            return None
        outs.append(max(sol.y[IDX_Y1, -1], 0.0))
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
    p0 = rng.uniform(0.0, 2.0, size=6)  # all positive
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
def plot_results(params, title="Full Cherry-Qian WTA XOR (grounded)"):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    t_common = np.linspace(0.0, T_FINAL, 500)
    for idx, (x1, x2) in enumerate(INPUTS):
        sol = simulate(x1, x2, params)
        S1 = sol.sol(t_common)[IDX_S1]
        S2 = sol.sol(t_common)[IDX_S2]
        Y1 = sol.sol(t_common)[IDX_Y1]
        Y2 = sol.sol(t_common)[IDX_Y2]
        axes[idx].plot(t_common, S1, label='S1', alpha=0.7)
        axes[idx].plot(t_common, S2, label='S2', alpha=0.7)
        axes[idx].plot(t_common, Y1, label='Y1 (output)', linewidth=2)
        axes[idx].plot(t_common, Y2, label='Y2', alpha=0.7)
        axes[idx].axhline(TARGETS[idx], color='r', linestyle='--', label='target')
        axes[idx].set_title(f"X1={x1}, X2={x2}")
        axes[idx].set_xlabel('time')
        axes[idx].set_ylabel('concentration')
        axes[idx].legend()
        axes[idx].grid(True)
    fig.suptitle(title)
    plt.tight_layout()
    fname = 'cherry_qian_wta_xor_grounded_results.png'
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
