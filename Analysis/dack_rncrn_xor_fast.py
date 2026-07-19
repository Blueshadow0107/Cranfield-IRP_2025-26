"""
Fast Dack-style RNCRN XOR demo: single seed, robust loss, quick result.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

N_EXEC = 3
N_PERC = 3
T_FINAL = 20.0
GAMMA = 0.05
DELTA = 1.0
SEED = 0

INPUTS = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
TARGETS = np.array([0.0, 1.0, 1.0, 0.0])

def sigma_gamma(z, gamma=GAMMA):
    return 0.5 * (z + np.sqrt(z**2 + 4.0*gamma))

def pack_params(alpha, omega, theta, beta3):
    return np.concatenate([alpha[2, :].ravel(), omega.ravel(), theta.ravel(), [beta3]])

def unpack_params(p):
    alpha = np.zeros((N_EXEC, N_PERC))
    alpha[2, :] = p[0:N_PERC]
    omega = p[N_PERC:N_PERC+N_PERC*N_EXEC].reshape((N_PERC, N_EXEC))
    theta = p[N_PERC+N_PERC*N_EXEC:N_PERC+N_PERC*N_EXEC+N_PERC]
    beta3 = p[-1]
    return alpha, omega, theta, beta3

def reduced_ode(t, X, alpha, omega, theta, beta3):
    Z = omega @ X + theta
    Y = sigma_gamma(Z)
    dX = np.zeros_like(X)
    dX[2] = beta3 - DELTA * X[2] + X[2] * (alpha[2, :] @ Y)
    return dX

def simulate(x1, x2, params, t_final=T_FINAL):
    alpha, omega, theta, beta3 = unpack_params(params)
    X0 = np.array([x1, x2, 0.0])
    sol = solve_ivp(
        reduced_ode, [0.0, t_final], X0,
        args=(alpha, omega, theta, beta3),
        method='LSODA', rtol=1e-6, atol=1e-8, dense_output=True,
    )
    return sol

def predict(params):
    outs = []
    for (x1, x2) in INPUTS:
        sol = simulate(x1, x2, params)
        if not sol.success:
            return None
        outs.append(sol.y[2, -1])
    return np.array(outs)

def loss(params):
    preds = predict(params)
    if preds is None:
        return 1e6
    return np.mean((preds - TARGETS)**2)

# Initialize
rng = np.random.default_rng(SEED)
p0 = rng.normal(0.0, 1.0, size=N_PERC + N_PERC*N_EXEC + N_PERC + 1)
print("Initial loss:", loss(p0))
print("Initial predictions:", predict(p0))

# Optimize with Nelder-Mead (derivative-free, robust)
result = minimize(
    loss, p0,
    method='Nelder-Mead',
    options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': True},
)

p_opt = result.x
print("\nOptimized loss:", result.fun)
print("Optimized predictions:", predict(p_opt))

alpha, omega, theta, beta3 = unpack_params(p_opt)
print("\nalpha (output row):", alpha[2, :])
print("omega:\n", omega)
print("theta:", theta)
print("beta3:", beta3)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.ravel()
t_common = np.linspace(0.0, T_FINAL, 500)
for idx, (x1, x2) in enumerate(INPUTS):
    sol = simulate(x1, x2, p_opt)
    X3 = sol.sol(t_common)[2]
    axes[idx].plot(t_common, X3, label='X3 (output)')
    axes[idx].axhline(TARGETS[idx], color='r', linestyle='--', label='target')
    axes[idx].set_title(f"X1={x1}, X2={x2}")
    axes[idx].set_xlabel('time')
    axes[idx].set_ylabel('concentration')
    axes[idx].legend()
    axes[idx].grid(True)
fig.suptitle('Dack-style RNCRN solving XOR')
plt.tight_layout()
plt.savefig('dack_rncrn_xor_fast_results.png', dpi=150)
print("Saved plot to dack_rncrn_xor_fast_results.png")
