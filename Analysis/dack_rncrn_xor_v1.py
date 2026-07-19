"""
Dack-style Recurrent Neural Chemical Reaction Network (RNCRN) for XOR.

Governing equations (mass-action ODEs):

  dX_i/dt = beta_i + X_i * sum_j alpha_{i,j} * Y_j
  dY_j/dt = (1/mu) * [ gamma + theta_j * Y_j
                       + Y_j * sum_i omega_{j,i} * X_i
                       - Y_j**2 ]

X = [X1, X2, X3]  : executive species
Y = [Y1, Y2]      : chemical perceptrons

X1, X2 are pinned inputs (beta=0, alpha rows zeroed).
X3 is the output.

All trainable parameters are non-negative for this first trial.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Architecture
# -----------------------------------------------------------------------------
N_EXEC = 3    # X1 (input), X2 (input), X3 (output)
N_PERC = 2    # Y1, Y2
T_FINAL = 10.0
MU = 0.01
GAMMA = 0.05

# Input/output data for XOR
INPUTS = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])
TARGETS = np.array([0.0, 1.0, 1.0, 0.0])

# Small positive init for perceptrons
Y0 = np.array([0.1, 0.1])

# -----------------------------------------------------------------------------
# Parameter packing/unpacking
# -----------------------------------------------------------------------------
def pack_params(alpha, omega, theta, beta3):
    """Flatten trainable parameters into a 1D vector."""
    # alpha is N_EXEC x N_PERC, but rows for X1 and X2 are forced to zero.
    # We only train alpha[2, :] (output row).
    return np.concatenate([
        alpha[2, :].ravel(),
        omega.ravel(),
        theta.ravel(),
        [beta3],
    ])

def unpack_params(p):
    """Reconstruct parameters from 1D vector."""
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

def clip_params(p):
    """Keep parameters non-negative."""
    return np.maximum(p, 0.0)

# -----------------------------------------------------------------------------
# ODE definition
# -----------------------------------------------------------------------------
def rncrn_ode(t, state, alpha, omega, theta, beta3, gamma, mu):
    X = state[:N_EXEC]
    Y = state[N_EXEC:]
    beta = np.array([0.0, 0.0, beta3])
    dX = beta + X * (alpha @ Y)
    dY = (1.0/mu) * (gamma + theta*Y + Y*(omega @ X) - Y**2)
    return np.concatenate([dX, dY])

def simulate(x1, x2, params, t_final=T_FINAL):
    """Integrate the RNCRN for one input pair. Return full solution."""
    alpha, omega, theta, beta3 = unpack_params(params)
    X0 = np.array([x1, x2, 0.0])
    state0 = np.concatenate([X0, Y0])
    sol = solve_ivp(
        rncrn_ode,
        [0.0, t_final],
        state0,
        args=(alpha, omega, theta, beta3, GAMMA, MU),
        method='LSODA',
        rtol=1e-7,
        atol=1e-9,
        dense_output=True,
    )
    return sol

def predict(params):
    """Predict X3(T) for all four XOR inputs."""
    outs = []
    for (x1, x2) in INPUTS:
        sol = simulate(x1, x2, params)
        outs.append(sol.y[2, -1])
    return np.array(outs)

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def loss(params):
    preds = predict(params)
    return np.mean((preds - TARGETS)**2)

def train(seed=42):
    rng = np.random.default_rng(seed)
    # Initial guess: small positive random values
    p0 = rng.uniform(0.0, 0.5, size=N_PERC + N_PERC*N_EXEC + N_PERC + 1)
    p0 = clip_params(p0)

    print("Initial parameters:", p0)
    print("Initial predictions:", predict(p0))
    print("Initial loss:", loss(p0))

    # Nelder-Mead is derivative-free and handles bounds poorly, so we just clip
    # inside the objective. For a cleaner constrained problem we could use L-BFGS-B.
    result = minimize(
        lambda q: loss(clip_params(q)),
        p0,
        method='Nelder-Mead',
        options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': True},
    )
    p_opt = clip_params(result.x)
    print("\nOptimized parameters:", p_opt)
    print("Optimized predictions:", predict(p_opt))
    print("Optimized loss:", loss(p_opt))
    return p_opt

# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
def plot_results(params, title="RNCRN XOR"):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    t_common = np.linspace(0.0, T_FINAL, 500)
    for idx, (x1, x2) in enumerate(INPUTS):
        sol = simulate(x1, x2, params)
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
    plt.savefig('dack_rncrn_xor_v1_results.png', dpi=150)
    print("Saved plot to dack_rncrn_xor_v1_results.png")

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    p_opt = train()
    print("\nFinal XOR truth table:")
    preds = predict(p_opt)
    for (x1, x2), target, pred in zip(INPUTS, TARGETS, preds):
        print(f"  X1={x1}, X2={x2} -> target={target}, predicted={pred:.4f}")
    plot_results(p_opt)
