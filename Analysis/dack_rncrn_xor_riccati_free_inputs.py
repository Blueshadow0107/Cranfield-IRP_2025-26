"""
Full two-timescale RCN for XOR without QSSA and with free inputs.

Each executive species X_i evolves freely from its initial condition:
    dX_i/dt = beta_i - delta_i * X_i + X_i * sum_j alpha_{ij} * Y_j

Each perceptron Y_j is fast:
    mu * dY_j/dt = gamma + a_j * Y_j - Y_j**2
    a_j = theta_j + sum_i omega_{ji} * X_i

The perceptron equation has an exact Riccati solution when X is frozen.
We use a first-order operator-splitting reaction step:
  1. Update Y_j via Riccati solve over dt with X fixed at X^n.
  2. Update X_i via a point-implicit (Patankar-style) Euler step with Y fixed at Y^{n+1}.

The inputs X1 and X2 are not pinned; their initial conditions encode the
input bits and they decay slowly.  The output is read from X3 at final time.

A saturating quadratic degradation term is added to X3 to prevent unbounded
growth while keeping the dynamics positive and mass-action-like.
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
N_EXEC = 3          # X1, X2 = inputs, X3 = output
N_PERC = 4
T_FINAL = 20.0
DT = 0.01
N_STEPS = int(T_FINAL / DT)
GAMMA = 0.05
MU = 0.02

# Executive degradation: small for inputs so they persist, larger for output
DELTA = np.array([0.05, 0.05, 1.0])

# Saturating degradation on X3: -K_SAT * X3**2
K_SAT = 1.0

INPUTS = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
TARGETS = np.array([0.0, 1.0, 1.0, 0.0])

# Parameter clipping used inside the loss/simulation for robustness
PARAM_CLIP_OMEGA = 2.0
PARAM_CLIP_ALPHA = 2.0
PARAM_CLIP_THETA = 2.0
PARAM_CLIP_BETA = 2.0

# -----------------------------------------------------------------------------
def sigma_gamma(z, gamma=GAMMA):
    zc = np.clip(z, -50.0, 50.0)
    return 0.5 * (zc + np.sqrt(zc**2 + 4.0*gamma))

def riccati_solve(y0, a, gamma, mu, dt):
    """Exact solution of mu * dy/dt = gamma + a*y - y**2 over dt."""
    a = np.clip(a, -50.0, 50.0)
    disc = np.sqrt(a**2 + 4.0 * gamma)
    y_plus = 0.5 * (a + disc)
    y_minus = 0.5 * (a - disc)

    if np.abs(y0 - y_plus) < 1e-14:
        return y_plus

    lam = disc / mu
    exp_term = np.exp(-lam * dt)
    numerator = y_plus * (y0 - y_minus) - y_minus * (y0 - y_plus) * exp_term
    denominator = (y0 - y_minus) - (y0 - y_plus) * exp_term

    if np.abs(denominator) < 1e-30:
        return y_plus

    return numerator / denominator

def pack_params(alpha, omega, theta, beta_out):
    return np.concatenate([
        alpha.ravel(),          # N_EXEC * N_PERC
        omega.ravel(),          # N_PERC * N_EXEC
        theta.ravel(),          # N_PERC
        [beta_out],             # 1
    ])

def unpack_params(p):
    alpha = p[0:N_EXEC*N_PERC].reshape((N_EXEC, N_PERC))
    omega = p[N_EXEC*N_PERC:N_EXEC*N_PERC + N_PERC*N_EXEC].reshape((N_PERC, N_EXEC))
    theta = p[N_EXEC*N_PERC + N_PERC*N_EXEC:N_EXEC*N_PERC + N_PERC*N_EXEC + N_PERC]
    beta_out = p[-1]
    return alpha, omega, theta, beta_out

def clip_params(p):
    """Clip parameters to the ranges used during simulation."""
    alpha, omega, theta, beta_out = unpack_params(p)
    omega = np.clip(omega, -PARAM_CLIP_OMEGA, PARAM_CLIP_OMEGA)
    alpha = np.clip(alpha, -PARAM_CLIP_ALPHA, PARAM_CLIP_ALPHA)
    theta = np.clip(theta, -PARAM_CLIP_THETA, PARAM_CLIP_THETA)
    beta_out = np.clip(beta_out, 0.0, PARAM_CLIP_BETA)
    return pack_params(alpha, omega, theta, beta_out)

# -----------------------------------------------------------------------------
def simulate_reaction(x_init, params, n_steps=N_STEPS, dt=DT, mu=MU, gamma=GAMMA,
                      k_sat=K_SAT):
    """Integrate the reaction dynamics (no diffusion)."""
    params = clip_params(params)
    alpha, omega, theta, beta_out = unpack_params(params)
    beta = np.array([0.0, 0.0, beta_out])

    X = x_init.copy()
    # Initialize perceptrons on the quasi-static manifold for a smooth start
    a = np.clip(omega @ X + theta, -50.0, 50.0)
    Y = sigma_gamma(a, gamma)

    X_traj = np.zeros((n_steps + 1, N_EXEC))
    Y_traj = np.zeros((n_steps + 1, N_PERC))
    X_traj[0] = X
    Y_traj[0] = Y

    for n in range(n_steps):
        # --- Perceptron substep (Riccati solve, X frozen) ---
        a = np.clip(omega @ X + theta, -50.0, 50.0)
        for j in range(N_PERC):
            Y[j] = riccati_solve(Y[j], a[j], gamma, mu, dt)

        # --- Executive substep (point-implicit Patankar-style, Y frozen) ---
        c = alpha @ Y          # N_EXEC, net coefficient of X_i
        for i in range(N_EXEC - 1):
            prod_rate = beta[i] + max(c[i], 0.0) * X[i]
            dest_rate = DELTA[i] + max(-c[i], 0.0)
            X[i] = (X[i] + dt * prod_rate) / (1.0 + dt * dest_rate)

        # Output species with saturating degradation (semi-implicit on X3^2)
        i = N_EXEC - 1
        prod_rate = beta[i] + max(c[i], 0.0) * X[i]
        dest_rate = DELTA[i] + max(-c[i], 0.0)
        X[i] = (X[i] + dt * prod_rate) / (1.0 + dt * (dest_rate + k_sat * X[i]))

        X_traj[n + 1] = X
        Y_traj[n + 1] = Y

    return X, X_traj, Y_traj

def predict(params, n_steps=N_STEPS, dt=DT):
    outs = []
    for (x1, x2) in INPUTS:
        x_init = np.array([x1, x2, 0.05])
        X_final, _, _ = simulate_reaction(x_init, params, n_steps, dt)
        outs.append(X_final[2])
    return np.array(outs)

# -----------------------------------------------------------------------------
def loss(params):
    preds = predict(params)
    if np.any(~np.isfinite(preds)):
        return 1e6

    mse = np.mean((preds - TARGETS)**2)

    penalty = 0.0
    if np.any(preds < 0):
        penalty += 1e4
    if np.any(preds > 5.0):
        penalty += 1e4

    reg = 1e-3 * np.sum(params**2)

    return mse + penalty + reg

def train_one(seed=42):
    rng = np.random.default_rng(seed)
    n_params = N_EXEC*N_PERC + N_PERC*N_EXEC + N_PERC + 1
    p0 = rng.normal(0.0, 0.3, size=n_params)
    # Start with small feedback weights from X3 to perceptrons
    omega0 = p0[N_EXEC*N_PERC:N_EXEC*N_PERC + N_PERC*N_EXEC].reshape((N_PERC, N_EXEC))
    omega0[:, 2] = 0.0
    p0[N_EXEC*N_PERC:N_EXEC*N_PERC + N_PERC*N_EXEC] = omega0.ravel()

    bounds = [(-1.5, 1.5)] * (n_params - 1) + [(0.0, 2.0)]

    result = minimize(
        loss, p0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-7},
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
def plot_results(params, title="Full RCN with Riccati perceptrons (free inputs)"):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    t_common = np.linspace(0.0, T_FINAL, N_STEPS + 1)
    for idx, (x1, x2) in enumerate(INPUTS):
        x_init = np.array([x1, x2, 0.05])
        X_final, X_traj, Y_traj = simulate_reaction(x_init, params)
        axes[idx].plot(t_common, X_traj[:, 0], label='X1', alpha=0.7)
        axes[idx].plot(t_common, X_traj[:, 1], label='X2', alpha=0.7)
        axes[idx].plot(t_common, X_traj[:, 2], label='X3 (output)', linewidth=2)
        axes[idx].axhline(TARGETS[idx], color='r', linestyle='--', label='target')
        axes[idx].set_title(f"X1(0)={x1}, X2(0)={x2}")
        axes[idx].set_xlabel('time')
        axes[idx].set_ylabel('concentration')
        axes[idx].legend()
        axes[idx].grid(True)
    fig.suptitle(title)
    plt.tight_layout()
    fname = 'dack_rncrn_xor_riccati_free_inputs_results.png'
    plt.savefig(fname, dpi=150)
    print(f"Saved plot to {fname}")

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    p_opt = train(n_restarts=10)
    if p_opt is not None:
        preds = predict(p_opt)
        print("\nFinal XOR truth table:")
        for (x1, x2), target, pred in zip(INPUTS, TARGETS, preds):
            print(f"  X1(0)={x1}, X2(0)={x2} -> target={target}, predicted={pred:.4f}")
        plot_results(p_opt)
