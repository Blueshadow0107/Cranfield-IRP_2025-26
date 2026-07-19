"""
Recurrent Dack-style RNCRN for XOR with bounded output.

We transform the output variable X3 through a sigmoid:

  X3 = sigmoid(u) = 1 / (1 + exp(-u))

so X3 stays in (0, 1).  The perceptrons depend on the actual output X3,
creating the recurrent feedback loop.  We integrate in u-space:

  du/dt = [ beta3 - delta*X3 + X3 * sum_j alpha_j * Y_j* ] / [X3 * (1 - X3)]

with a small floor on the denominator for numerical stability.

The network is recurrent because X3(t) feeds back into the perceptron inputs
and therefore influences its own future evolution.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
N_PERC = 4
T_FINAL = 20.0
GAMMA = 0.05
DELTA = 1.0
EPS = 1e-4         # floor for denominator
SEED = 0

INPUTS = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
TARGETS = np.array([0.0, 1.0, 1.0, 0.0])

# -----------------------------------------------------------------------------
def sigmoid(u):
    return 1.0 / (1.0 + np.exp(-u))

def sigma_gamma(z, gamma=GAMMA):
    zc = np.clip(z, -50.0, 50.0)
    return 0.5 * (zc + np.sqrt(zc**2 + 4.0*gamma))

def pack_params(alpha, omega, theta, beta3):
    return np.concatenate([alpha.ravel(), omega.ravel(), theta.ravel(), [beta3]])

def unpack_params(p):
    alpha = p[0:N_PERC]
    omega = p[N_PERC:N_PERC+3*N_PERC].reshape((N_PERC, 3))
    theta = p[N_PERC+3*N_PERC:N_PERC+3*N_PERC+N_PERC]
    beta3 = p[-1]
    return alpha, omega, theta, beta3

# -----------------------------------------------------------------------------
def perceptrons(x1, x2, x3, omega, theta):
    z = omega @ np.array([x1, x2, x3]) + theta
    return sigma_gamma(z)

def recurrent_ode(t, u, x1, x2, alpha, omega, theta, beta3):
    x3 = sigmoid(u[0])
    Y = perceptrons(x1, x2, x3, omega, theta)
    dx3dt = beta3 - DELTA*x3 + x3 * (alpha @ Y)
    denom = max(x3 * (1.0 - x3), EPS)
    return np.array([dx3dt / denom])

def simulate(x1, x2, params, t_final=T_FINAL):
    alpha, omega, theta, beta3 = unpack_params(params)
    # Initial u corresponding to X3 = 0.05 (avoid exact 0)
    u0 = np.log(0.05 / 0.95)
    sol = solve_ivp(
        recurrent_ode, [0.0, t_final], [u0],
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
        outs.append(sigmoid(sol.y[0, -1]))
    return np.array(outs)

# -----------------------------------------------------------------------------
def loss(params):
    preds = predict(params)
    if preds is None:
        return 1e6
    return np.mean((preds - TARGETS)**2)

def train_one(seed=42):
    rng = np.random.default_rng(seed)
    # Initialize feedback weights (omega[:,2]) small, feedforward weights moderate
    p0 = rng.normal(0.0, 0.5, size=N_PERC + 3*N_PERC + N_PERC + 1)
    # Zero out the feedback initially for stability
    p0[N_PERC+2*N_PERC:N_PERC+3*N_PERC] = 0.0
    bounds = [(-2.0, 2.0)] * len(p0)
    result = minimize(
        loss, p0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-7},
    )
    return result.x, result.fun

def train(n_restarts=5):
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
def plot_results(params, title="Recurrent RNCRN with bounded output solving XOR"):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    t_common = np.linspace(0.0, T_FINAL, 500)
    for idx, (x1, x2) in enumerate(INPUTS):
        sol = simulate(x1, x2, params)
        X3 = sigmoid(sol.sol(t_common)[0])
        axes[idx].plot(t_common, X3, label='X3 (output)')
        axes[idx].axhline(TARGETS[idx], color='r', linestyle='--', label='target')
        axes[idx].set_title(f"X1={x1}, X2={x2}")
        axes[idx].set_xlabel('time')
        axes[idx].set_ylabel('concentration')
        axes[idx].legend()
        axes[idx].grid(True)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig('dack_rncrn_xor_recurrent_bounded_results.png', dpi=150)
    print("Saved plot to dack_rncrn_xor_recurrent_bounded_results.png")

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    p_opt = train(n_restarts=5)
    if p_opt is not None:
        preds = predict(p_opt)
        print("\nFinal XOR truth table:")
        for (x1, x2), target, pred in zip(INPUTS, TARGETS, preds):
            print(f"  X1={x1}, X2={x2} -> target={target}, predicted={pred:.4f}")
        plot_results(p_opt)

        # Bistability / history-dependence check
        print("\nBistability check for X1=1, X2=0:")
        alpha, omega, theta, beta3 = unpack_params(p_opt)
        for x3_init in [0.05, 0.2, 0.5, 0.8, 0.95]:
            u0 = np.log(x3_init / (1.0 - x3_init))
            sol = solve_ivp(
                recurrent_ode, [0.0, T_FINAL], [u0],
                args=(1.0, 0.0, alpha, omega, theta, beta3),
                method='LSODA', rtol=1e-6, atol=1e-8,
            )
            print(f"  X3(0)={x3_init} -> X3(T)={sigmoid(sol.y[0, -1]):.4f}")
