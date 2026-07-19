"""
2D reaction-diffusion RCN for XOR with pinned inputs — FFT-accelerated.

X1 and X2 are fixed spatial input patterns (not dynamic).
X3 is the only dynamic executive species.
Y_j are fast perceptrons with 2D diffusion.

PDEs:
    ∂X3/∂t = D_X3 ∇²X3 + β_out - δ_3 X3 + X3 Σ_j α_3j Y_j - κ X3²
    ∂Y_j/∂t = D_Y ∇²Y_j + (1/μ)(γ + a_j Y_j - Y_j²)
    a_j = θ_j + ω_1j X1 + ω_2j X2 + ω_3j X3

Numerics:
    - Operator splitting: diffusion step, then reaction step.
    - Spectral (FFT) diffusion step — exact and unconditionally stable.
    - Riccati solve for perceptrons (exact for frozen X).
    - Patankar-style point-implicit step for X3.

Input encoding for XOR:
    X1 = 1 on the left half of the domain, 0 on the right.
    X2 = 1 on the bottom half of the domain, 0 on the top.
    Output is read as the spatial average of X3 over a central probe region.
    Targets: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0.
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Grid and time
NX, NY = 32, 32
DX = 0.2
DT = 0.01
T_FINAL = 5.0
N_STEPS = int(T_FINAL / DT)

# Precompute spectral diffusion kernel
KX = 2.0 * np.pi * np.fft.fftfreq(NX, d=DX)[:, None]
KY = 2.0 * np.pi * np.fft.fftfreq(NY, d=DX)[None, :]
K2 = KX ** 2 + KY ** 2
DIFF_KERNEL_X3 = np.exp(-0.02 * K2 * DT)  # D_X3 = 0.02
DIFF_KERNEL_Y = np.exp(-0.05 * K2 * DT)   # D_Y = 0.05

# Diffusion coefficients (kept for reference / re-derivation)
D_X3 = 0.02
D_Y = 0.05

# Perceptrons
N_PERC = 4
GAMMA = 0.05
MU = 0.02

# Executive
DELTA_3 = 1.0
K_SAT = 1.0

# Parameter clipping
PARAM_CLIP = 2.0

# Probe region for scalar readout (central patch)
PROBE_X = slice(NX // 2 - 4, NX // 2 + 4)
PROBE_Y = slice(NY // 2 - 4, NY // 2 + 4)

INPUTS = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
TARGETS = np.array([0.0, 1.0, 1.0, 0.0])

# -----------------------------------------------------------------------------
def make_input_fields(x1, x2, nx=NX, ny=NY):
    """Create pinned 2D input patterns."""
    X1 = np.zeros((nx, ny))
    X2 = np.zeros((nx, ny))
    if x1 > 0.5:
        X1[:nx // 2, :] = 1.0
    if x2 > 0.5:
        X2[:, :ny // 2] = 1.0
    return X1, X2


def diffusion_step_spectral(C, kernel):
    """Exact diffusion step using FFT (periodic boundary conditions)."""
    C_hat = np.fft.fft2(C)
    C_hat *= kernel
    return np.fft.ifft2(C_hat).real


def sigma_gamma(z, gamma=GAMMA):
    zc = np.clip(z, -50.0, 50.0)
    return 0.5 * (zc + np.sqrt(zc ** 2 + 4.0 * gamma))


def riccati_solve(y0, a, gamma, mu, dt):
    """Exact solution of mu * dy/dt = gamma + a*y - y**2 over dt."""
    a = np.clip(a, -50.0, 50.0)
    disc = np.sqrt(a ** 2 + 4.0 * gamma)
    y_plus = 0.5 * (a + disc)
    y_minus = 0.5 * (a - disc)

    lam = disc / mu
    exp_term = np.exp(-lam * dt)
    denom = (y0 - y_minus) - (y0 - y_plus) * exp_term

    y_new = np.where(
        np.abs(denom) < 1e-30,
        y_plus,
        (y_plus * (y0 - y_minus) - y_minus * (y0 - y_plus) * exp_term) / denom
    )
    return y_new


# -----------------------------------------------------------------------------
def pack_params(alpha_3, omega, theta, beta_out):
    return np.concatenate([alpha_3.ravel(), omega.ravel(), theta.ravel(), [beta_out]])


def unpack_params(p):
    n = N_PERC
    alpha_3 = p[0:n]
    omega = p[n:n + 3 * n].reshape((n, 3))
    theta = p[n + 3 * n:n + 3 * n + n]
    beta_out = p[-1]
    return alpha_3, omega, theta, beta_out


def clip_params(p):
    alpha_3, omega, theta, beta_out = unpack_params(p)
    alpha_3 = np.clip(alpha_3, -PARAM_CLIP, PARAM_CLIP)
    omega = np.clip(omega, -PARAM_CLIP, PARAM_CLIP)
    theta = np.clip(theta, -PARAM_CLIP, PARAM_CLIP)
    beta_out = np.clip(beta_out, 0.0, PARAM_CLIP)
    return pack_params(alpha_3, omega, theta, beta_out)


# -----------------------------------------------------------------------------
def simulate_2d(x1, x2, params, n_steps=N_STEPS, dt=DT, mu=MU, gamma=GAMMA,
                k_sat=K_SAT):
    """Integrate the 2D reaction-diffusion RCN."""
    params = clip_params(params)
    alpha_3, omega, theta, beta_out = unpack_params(params)

    X1, X2 = make_input_fields(x1, x2)
    X3 = np.full((NX, NY), 0.05)

    # Initialize perceptrons on quasi-static manifold
    a = np.clip(
        omega[:, 0][:, None, None] * X1 +
        omega[:, 1][:, None, None] * X2 +
        omega[:, 2][:, None, None] * X3 +
        theta[:, None, None],
        -50.0, 50.0
    )
    Y = sigma_gamma(a, gamma)

    for n in range(n_steps):
        # --- Diffusion substep (spectral, exact) ---
        X3 = diffusion_step_spectral(X3, DIFF_KERNEL_X3)
        Y = np.array([diffusion_step_spectral(Y[j], DIFF_KERNEL_Y) for j in range(N_PERC)])

        # --- Reaction substep (pointwise) ---
        a = np.clip(
            omega[:, 0][:, None, None] * X1 +
            omega[:, 1][:, None, None] * X2 +
            omega[:, 2][:, None, None] * X3 +
            theta[:, None, None],
            -50.0, 50.0
        )
        for j in range(N_PERC):
            Y[j] = riccati_solve(Y[j], a[j], gamma, mu, dt)

        # X3 update (Patankar-style with saturating degradation)
        c = np.tensordot(alpha_3, Y, axes=([0], [0]))  # (NX, NY)
        prod_rate = beta_out + np.maximum(c, 0.0) * X3
        dest_rate = DELTA_3 + np.maximum(-c, 0.0)
        X3 = (X3 + dt * prod_rate) / (1.0 + dt * (dest_rate + k_sat * X3))

        # Hard ceiling for safety
        X3 = np.clip(X3, 0.0, 5.0)
        Y = np.clip(Y, 0.0, 10.0)

    return X3, Y, X1, X2


def predict(params, n_steps=N_STEPS, dt=DT):
    outs = []
    for (x1, x2) in INPUTS:
        X3, _, _, _ = simulate_2d(x1, x2, params, n_steps, dt)
        outs.append(np.mean(X3[PROBE_X, PROBE_Y]))
    return np.array(outs)


# -----------------------------------------------------------------------------
def loss(params):
    preds = predict(params)
    if np.any(~np.isfinite(preds)):
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
    n_params = N_PERC + 3 * N_PERC + N_PERC + 1
    p0 = rng.normal(0.0, 0.3, size=n_params)
    # Zero feedback from X3 to perceptrons initially
    omega0 = p0[N_PERC:N_PERC + 3 * N_PERC].reshape((N_PERC, 3))
    omega0[:, 2] = 0.0
    p0[N_PERC:N_PERC + 3 * N_PERC] = omega0.ravel()

    bounds = [(-PARAM_CLIP, PARAM_CLIP)] * (n_params - 1) + [(0.0, PARAM_CLIP)]

    result = minimize(
        loss, p0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-7},
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
def plot_results(params, title="2D RCN with pinned inputs (spectral diffusion)"):
    fig, axes = plt.subplots(4, 4, figsize=(14, 12))
    for idx, (x1, x2) in enumerate(INPUTS):
        X3, Y, X1, X2 = simulate_2d(x1, x2, params)

        ax = axes[idx, 0]
        im = ax.imshow(X1.T, origin='lower', cmap='Blues', vmin=0, vmax=1)
        ax.set_title(f"X1 (input {x1})")
        plt.colorbar(im, ax=ax)

        ax = axes[idx, 1]
        im = ax.imshow(X2.T, origin='lower', cmap='Oranges', vmin=0, vmax=1)
        ax.set_title(f"X2 (input {x2})")
        plt.colorbar(im, ax=ax)

        ax = axes[idx, 2]
        im = ax.imshow(X3.T, origin='lower', cmap='Greens')
        ax.set_title(f"X3 output (target {TARGETS[idx]})")
        plt.colorbar(im, ax=ax)

        ax = axes[idx, 3]
        im = ax.imshow(Y[0].T, origin='lower', cmap='Purples')
        ax.set_title("Y1")
        plt.colorbar(im, ax=ax)

    fig.suptitle(title)
    plt.tight_layout()
    fname = 'dack_rncrn_xor_2d_pinned_inputs_results.png'
    plt.savefig(fname, dpi=150)
    print(f"Saved plot to {fname}")


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    print(f"Grid: {NX}x{NY}, DT={DT}, N_STEPS={N_STEPS}")
    p_opt = train(n_restarts=3)
    if p_opt is not None:
        preds = predict(p_opt)
        print("\nFinal XOR truth table:")
        for (x1, x2), target, pred in zip(INPUTS, TARGETS, preds):
            print(f"  X1={x1}, X2={x2} -> target={target}, predicted={pred:.4f}")
        plot_results(p_opt)
