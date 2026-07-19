"""
Full recurrent Dack-style RNCRN for XOR.

Perceptrons depend on all executive species, including the output X3:

  Y_j* = sigma_gamma( omega_{j1}*X1 + omega_{j2}*X2 + omega_{j3}*X3 + theta_j )

Output dynamics:

  dX3/dt = beta3 - delta*X3 + X3 * sum_j alpha_j * Y_j*(X1, X2, X3)

This creates a feedback loop: X3 affects the perceptrons, and the perceptrons
affect the growth/degradation of X3.  The system can exhibit bistability,
hysteresis, or switching depending on parameters.

X1 and X2 are pinned inputs.  We train the network so that X3(T) matches the
XOR truth table.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Architecture and hyperparameters
# -----------------------------------------------------------------------------
N_PERC = 4          # number of chemical perceptrons
T_FINAL = 20.0      # readout time
GAMMA = 0.05        # smoothness of chemical ReLU
DELTA = 1.0         # output degradation rate
SEED = 0            # random seed for reproducibility

INPUTS = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
TARGETS = np.array([0.0, 1.0, 1.0, 0.0])

# -----------------------------------------------------------------------------
def sigma_gamma(z, gamma=GAMMA):
    """Chemical smooth ReLU activation."""
    return 0.5 * (z + np.sqrt(z**2 + 4.0*gamma))

# -----------------------------------------------------------------------------
# Parameter packing/unpacking
# Parameters: alpha (N_PERC), omega (N_PERC, 3), theta (N_PERC), beta3
# -----------------------------------------------------------------------------
def pack_params(alpha, omega, theta, beta3):
    return np.concatenate([alpha.ravel(), omega.ravel(), theta.ravel(), [beta3]])

def unpack_params(p):
    alpha = p[0:N_PERC]
    omega = p[N_PERC:N_PERC+3*N_PERC].reshape((N_PERC, 3))
    theta = p[N_PERC+3*N_PERC:N_PERC+3*N_PERC+N_PERC]
    beta3 = p[-1]
    return alpha, omega, theta, beta3

# -----------------------------------------------------------------------------
# Recurrent output ODE
# -----------------------------------------------------------------------------
def perceptrons(x1, x2, x3, omega, theta):
    z = omega @ np.array([x1, x2, x3]) + theta
    return sigma_gamma(z)

def recurrent_ode(t, X3, x1, x2, alpha, omega, theta, beta3):
    Y = perceptrons(x1, x2, X3[0], omega, theta)
    return np.array([beta3 - DELTA * X3[0] + X3[0] * (alpha @ Y)])

def simulate(x1, x2, params, t_final=T_FINAL):
    alpha, omega, theta, beta3 = unpack_params(params)
    sol = solve_ivp(
        recurrent_ode, [0.0, t_final], [0.0],
        args=(x1, x2, alpha, omega, theta, beta3),
        method='LSODA', rtol=1e-6, atol=1e-8, dense_output=True,
    )
    return sol

def predict(params):
    outs = []
    for (x1, x2) in INPUTS:
        sol = simulate(x1, x2, params)
        if not sol.success:
            return None
        outs.append(sol.y[0, -1])
    return np.array(outs)

# -----------------------------------------------------------------------------
# Training with robust failure handling
# -----------------------------------------------------------------------------
def loss(params):
    preds = predict(params)
    if preds is None:
        return 1e6
    return np.mean((preds - TARGETS)**2)

def train_one(seed=42):
    rng = np.random.default_rng(seed)
    # Smaller init to avoid blow-up from strong feedback
    p0 = rng.normal(0.0, 0.5, size=N_PERC + 3*N_PERC + N_PERC + 1)
    bounds = [(-3.0, 3.0)] * len(p0)
    result = minimize(
        loss, p0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8},
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
# Visualization
# -----------------------------------------------------------------------------
def plot_results(params, title="Full recurrent RNCRN solving XOR"):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    t_common = np.linspace(0.0, T_FINAL, 500)
    for idx, (x1, x2) in enumerate(INPUTS):
        sol = simulate(x1, x2, params)
        if not sol.success:
            axes[idx].text(0.5, 0.5, 'integration failed', ha='center')
            continue
        X3 = sol.sol(t_common)[0]
        axes[idx].plot(t_common, X3, label='X3 (output)')
        axes[idx].axhline(TARGETS[idx], color='r', linestyle='--', label='target')
        axes[idx].set_title(f"X1={x1}, X2={x2}")
        axes[idx].set_xlabel('time')
        axes[idx].set_ylabel('concentration')
        axes[idx].legend()
        axes[idx].grid(True)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig('dack_rncrn_xor_recurrent_results.png', dpi=150)
    print("Saved plot to dack_rncrn_xor_recurrent_results.png")

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    p_opt = train(n_restarts=10)
    if p_opt is not None:
        preds = predict(p_opt)
        print("\nFinal XOR truth table:")
        for (x1, x2), target, pred in zip(INPUTS, TARGETS, preds):
            print(f"  X1={x1}, X2={x2} -> target={target}, predicted={pred:.4f}")
        plot_results(p_opt)

        # Show bistability check: run from different X3(0) for one input
        print("\nBistability check for X1=1, X2=0:")
        alpha, omega, theta, beta3 = unpack_params(p_opt)
        for x3_init in [0.0, 0.5, 1.0, 2.0]:
            sol = solve_ivp(
                recurrent_ode, [0.0, T_FINAL], [x3_init],
                args=(1.0, 0.0, alpha, omega, theta, beta3),
                method='LSODA', rtol=1e-6, atol=1e-8,
            )
            print(f"  X3(0)={x3_init} -> X3(T)={sol.y[0, -1]:.4f}")
