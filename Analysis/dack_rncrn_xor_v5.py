"""
Dack-style RNCRN for XOR, quasi-static reduction, with output degradation.

Robust training version: failed ODE integrations return a large loss instead
of crashing, so random restarts can continue.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Architecture
# -----------------------------------------------------------------------------
N_EXEC = 3
N_PERC = 3
T_FINAL = 20.0
GAMMA = 0.05
DELTA = 1.0

INPUTS = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])
TARGETS = np.array([0.0, 1.0, 1.0, 0.0])

# -----------------------------------------------------------------------------
def sigma_gamma(z, gamma=GAMMA):
    return 0.5 * (z + np.sqrt(z**2 + 4.0*gamma))

# -----------------------------------------------------------------------------
def pack_params(alpha, omega, theta, beta3):
    return np.concatenate([
        alpha[2, :].ravel(),
        omega.ravel(),
        theta.ravel(),
        [beta3],
    ])

def unpack_params(p):
    alpha = np.zeros((N_EXEC, N_PERC))
    k = 0
    alpha[2, :] = p[k:k+N_PERC]
    k += N_PERC
    omega = p[k:k+N_PERC*N_EXEC].reshape((N_PERC, N_EXEC))
    k += N_PERC*N_EXEC
    theta = p[k:k+N_PERC]
    k += N_PERC
    beta3 = p[k]
    return alpha, omega, theta, beta3

# -----------------------------------------------------------------------------
def reduced_ode(t, X, alpha, omega, theta, beta3):
    Z = omega @ X + theta
    Y = sigma_gamma(Z)
    dX = np.zeros_like(X)
    dX[0] = 0.0
    dX[1] = 0.0
    dX[2] = beta3 - DELTA * X[2] + X[2] * (alpha[2, :] @ Y)
    return dX

def simulate(x1, x2, params, t_final=T_FINAL):
    alpha, omega, theta, beta3 = unpack_params(params)
    X0 = np.array([x1, x2, 0.0])
    sol = solve_ivp(
        reduced_ode,
        [0.0, t_final],
        X0,
        args=(alpha, omega, theta, beta3),
        method='RK45',
        rtol=1e-7,
        atol=1e-9,
        dense_output=True,
    )
    return sol

def predict(params):
    outs = []
    for (x1, x2) in INPUTS:
        sol = simulate(x1, x2, params)
        if not sol.success:
            return None  # signal failure
        outs.append(sol.y[2, -1])
    return np.array(outs)

# -----------------------------------------------------------------------------
def loss(params):
    preds = predict(params)
    if preds is None:
        return 1e6
    return np.mean((preds - TARGETS)**2)

def train_one(seed=42):
    rng = np.random.default_rng(seed)
    p0 = rng.normal(0.0, 1.0, size=N_PERC + N_PERC*N_EXEC + N_PERC + 1)
    bounds = [(-5.0, 5.0)] * len(p0)
    result = minimize(
        loss,
        p0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8},
    )
    return result.x, result.fun

def train(n_restarts=20):
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
def plot_results(params, title="RNCRN XOR (quasi-static, output degradation)"):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    t_common = np.linspace(0.0, T_FINAL, 500)
    for idx, (x1, x2) in enumerate(INPUTS):
        sol = simulate(x1, x2, params)
        if not sol.success:
            axes[idx].text(0.5, 0.5, 'integration failed', ha='center')
            continue
        X3 = sol.sol(t_common)[2]
        axes[idx].plot(t_common, X3, label='X3 (output)')
        axes[idx].axhline(TARGETS[idx], color='r', linestyle='--', label='target')
        axes[idx].set_title(f"X1={x1}, X2={x2}")
        axes[idx].set_xlabel('time')
        axes[idx].set_ylabel('concentration')
        axes[idx].legend()
        axes[idx].grid(True)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig('dack_rncrn_xor_v5_results.png', dpi=150)
    print("Saved plot to dack_rncrn_xor_v5_results.png")

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    p_opt = train(n_restarts=20)
    if p_opt is not None:
        preds = predict(p_opt)
        print("\nFinal XOR truth table:")
        for (x1, x2), target, pred in zip(INPUTS, TARGETS, preds):
            print(f"  X1={x1}, X2={x2} -> target={target}, predicted={pred:.4f}")
        plot_results(p_opt)
