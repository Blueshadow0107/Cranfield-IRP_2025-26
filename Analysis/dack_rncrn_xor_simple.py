"""
Dack-style RNCRN for XOR, simplified quasi-static reduction.

Perceptrons depend only on the pinned inputs X1, X2 (no feedback from X3).
The output X3 obeys:

  dX3/dt = beta3 - delta * X3 + X3 * sum_j alpha_j * Y_j*

with steady state (when it exists):

  X3* = beta3 / (delta - sum_j alpha_j * Y_j*)

Y_j* = sigma_gamma( omega_{j1} * X1 + omega_{j2} * X2 + theta_j )

This is a continuous nonlinear mapping from inputs to output, trained to
match the XOR truth table.
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
SEED = 0

INPUTS = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
TARGETS = np.array([0.0, 1.0, 1.0, 0.0])

def sigma_gamma(z, gamma=GAMMA):
    return 0.5 * (z + np.sqrt(z**2 + 4.0*gamma))

# Parameters: alpha (N_PERC), omega (N_PERC, 2), theta (N_PERC), beta3
def pack_params(alpha, omega, theta, beta3):
    return np.concatenate([alpha.ravel(), omega.ravel(), theta.ravel(), [beta3]])

def unpack_params(p):
    alpha = p[0:N_PERC]
    omega = p[N_PERC:N_PERC+2*N_PERC].reshape((N_PERC, 2))
    theta = p[N_PERC+2*N_PERC:N_PERC+2*N_PERC+N_PERC]
    beta3 = p[-1]
    return alpha, omega, theta, beta3

def perceptrons(x1, x2, omega, theta):
    z = omega @ np.array([x1, x2]) + theta
    return sigma_gamma(z)

def output_ode(t, X3, x1, x2, alpha, omega, theta, beta3):
    Y = perceptrons(x1, x2, omega, theta)
    return beta3 - DELTA * X3[0] + X3[0] * (alpha @ Y)

def simulate(x1, x2, params, t_final=T_FINAL):
    alpha, omega, theta, beta3 = unpack_params(params)
    sol = solve_ivp(
        output_ode, [0.0, t_final], [0.0],
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

def loss(params):
    preds = predict(params)
    if preds is None:
        return 1e6
    return np.mean((preds - TARGETS)**2)

# Initialize and optimize
rng = np.random.default_rng(SEED)
p0 = rng.normal(0.0, 1.0, size=N_PERC + 2*N_PERC + N_PERC + 1)
print("Initial loss:", loss(p0))
print("Initial predictions:", predict(p0))

result = minimize(
    loss, p0,
    method='Nelder-Mead',
    options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': True},
)

p_opt = result.x
print("\nOptimized loss:", result.fun)
print("Optimized predictions:", predict(p_opt))

alpha, omega, theta, beta3 = unpack_params(p_opt)
print("\nalpha:", alpha)
print("omega:\n", omega)
print("theta:", theta)
print("beta3:", beta3)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.ravel()
t_common = np.linspace(0.0, T_FINAL, 500)
for idx, (x1, x2) in enumerate(INPUTS):
    sol = simulate(x1, x2, p_opt)
    X3 = sol.sol(t_common)[0]
    axes[idx].plot(t_common, X3, label='X3 (output)')
    axes[idx].axhline(TARGETS[idx], color='r', linestyle='--', label='target')
    axes[idx].set_title(f"X1={x1}, X2={x2}")
    axes[idx].set_xlabel('time')
    axes[idx].set_ylabel('concentration')
    axes[idx].legend()
    axes[idx].grid(True)
fig.suptitle('Simplified Dack-style RNCRN solving XOR')
plt.tight_layout()
plt.savefig('dack_rncrn_xor_simple_results.png', dpi=150)
print("Saved plot to dack_rncrn_xor_simple_results.png")
