"""
Full two-timescale Dack RNCRN for XOR.

Implements the RNCRN from Dack et al. (2025) literally, using fast
perceptron reactions rather than an algebraic quasi-steady-state approximation.

Executive species: X3 (output)
Input species:    X1, X2 (pinned)
Perceptrons:      Y_j, j=1..M

Reactions:
  For executive species X3:
    ∅ --beta3--> X3
    X3 --delta--> ∅
    X3 + Y_j --alpha_j--> 2*X3 + Y_j   if alpha_j > 0
    X3 + Y_j --|alpha_j|--> Y_j        if alpha_j < 0

  For perceptron Y_j:
    ∅ --gamma/mu--> Y_j
    Y_j --1/mu--> ∅
    X_i + Y_j --omega_{j,i}/mu--> X_i + 2*Y_j   if omega_{j,i} > 0
    X_i + Y_j --|omega_{j,i}|/mu--> X_i         if omega_{j,i} < 0
    2*Y_j --1/mu--> Y_j

This gives the RREs:
    dX3/dt = beta3 - delta*X3 + X3 * sum_j alpha_j * Y_j
    mu * dY_j/dt = gamma + theta_j * Y_j
                   + Y_j * sum_i omega_{j,i} * X_i
                   - Y_j**2

Training strategy:
  1. Train the quasi-static (algebraic perceptron) approximation.
  2. Use the result to warm-start the full two-timescale system.
  3. Fine-tune with non-negativity clipping and perceptrons initialized on the
     quasi-static manifold.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import importlib.util
import sys

# Load the bounded quasi-static solver for warm-start
_spec = importlib.util.spec_from_file_location(
    "bounded", "/home/cyanidepopcorn/Cranfield/IndividualResearchProject/Analysis/dack_rncrn_xor_recurrent_bounded.py"
)
_bounded = importlib.util.module_from_spec(_spec)
sys.modules["bounded"] = _bounded
_spec.loader.exec_module(_bounded)

# -----------------------------------------------------------------------------
T_FINAL = 30.0
GAMMA = _bounded.GAMMA
DELTA_EXEC = _bounded.DELTA
MU = 0.02
SEED = 0

INPUTS = _bounded.INPUTS
TARGETS = _bounded.TARGETS

# -----------------------------------------------------------------------------
def sigma_gamma(z, gamma=GAMMA):
    zc = np.clip(z, -50.0, 50.0)
    return 0.5 * (zc + np.sqrt(zc**2 + 4.0*gamma))

def pack_params(alpha, omega, theta, beta3):
    return np.concatenate([alpha.ravel(), omega.ravel(), theta.ravel(), [beta3]])

def unpack_params(p, n_perc):
    alpha = p[0:n_perc]
    omega = p[n_perc:n_perc + 3*n_perc].reshape((n_perc, 3))
    theta = p[n_perc + 3*n_perc:n_perc + 3*n_perc + n_perc]
    beta3 = p[-1]
    return alpha, omega, theta, beta3

# -----------------------------------------------------------------------------
def rncrn_ode(t, state, x1, x2, alpha, omega, theta, beta3, mu, gamma):
    """Full two-timescale RNCRN ODE with non-negativity clipping."""
    x3 = max(state[0], 0.0)
    Y = np.maximum(state[1:], 0.0)

    dx3dt = beta3 - DELTA_EXEC * x3 + x3 * (alpha @ Y)
    X = np.array([x1, x2, x3])
    z = omega @ X + theta
    dYdt = (1.0 / mu) * (gamma + z * Y - Y**2)

    return np.concatenate([[dx3dt], dYdt])

def initial_perceptrons(x1, x2, x3_init, omega, theta):
    """Quasi-static perceptron values for initialization."""
    X = np.array([x1, x2, x3_init])
    return sigma_gamma(omega @ X + theta)

def simulate(x1, x2, params, n_perc, t_final=T_FINAL, mu=MU, gamma=GAMMA,
             x3_init=0.05):
    alpha, omega, theta, beta3 = unpack_params(params, n_perc)
    Y0 = initial_perceptrons(x1, x2, x3_init, omega, theta)
    state0 = np.concatenate([[x3_init], Y0])

    sol = solve_ivp(
        rncrn_ode, [0.0, t_final], state0,
        args=(x1, x2, alpha, omega, theta, beta3, mu, gamma),
        method='LSODA', rtol=1e-7, atol=1e-9, dense_output=True,
    )
    return sol

def predict(params, n_perc, t_final=T_FINAL, mu=MU, gamma=GAMMA):
    outs = []
    for (x1, x2) in INPUTS:
        sol = simulate(x1, x2, params, n_perc, t_final, mu, gamma)
        if not sol.success or np.any(~np.isfinite(sol.y)):
            return None
        outs.append(max(sol.y[0, -1], 0.0))
    return np.array(outs)

# -----------------------------------------------------------------------------
def loss(params, n_perc, t_final=T_FINAL, mu=MU, gamma=GAMMA):
    preds = predict(params, n_perc, t_final, mu, gamma)
    if preds is None:
        return 1e6
    penalty = 0.0
    if np.any(preds < 0):
        penalty += 1e4
    if np.any(preds > 20):
        penalty += 1e4
    return np.mean((preds - TARGETS)**2) + penalty

def train_full(n_perc, n_restarts_qs=5, n_restarts_full=3,
               t_final=T_FINAL, mu=MU, gamma=GAMMA):
    print(f"\n=== Training full RNCRN with M={n_perc} perceptrons, mu={mu} ===")

    # Stage 1: train quasi-static approximation using bounded module
    best_qs_loss = np.inf
    best_qs_params = None
    for seed in range(n_restarts_qs):
        p_opt, val = _bounded.train_one(seed)
        if val < best_qs_loss:
            best_qs_loss = val
            best_qs_params = p_opt
    print(f"  Quasi-static best loss: {best_qs_loss:.6f}")

    # Stage 2: fine-tune full system from warm start
    bounds = [(-3.0, 3.0)] * (len(best_qs_params) - 1) + [(0.0, 2.0)]
    best_full_loss = np.inf
    best_full_params = None
    for seed in range(n_restarts_full):
        # Small perturbation around quasi-static optimum
        rng = np.random.default_rng(seed)
        p0 = best_qs_params + rng.normal(0.0, 0.05, size=len(best_qs_params))
        p0 = np.clip(p0, [b[0] for b in bounds], [b[1] for b in bounds])
        try:
            res = minimize(
                lambda p: loss(p, n_perc, t_final, mu, gamma), p0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-7},
            )
        except Exception as e:
            print(f"  full restart {seed}: failed ({e})")
            continue
        print(f"  full restart {seed}: loss = {res.fun:.6f}")
        if res.fun < best_full_loss:
            best_full_loss = res.fun
            best_full_params = res.x
    print(f"Best full-system loss: {best_full_loss:.6f}")
    if best_full_params is not None:
        print(f"Best full-system predictions: {predict(best_full_params, n_perc, t_final, mu, gamma)}")
    return best_full_params, best_full_loss

# -----------------------------------------------------------------------------
def plot_results(params, n_perc, title_suffix="", t_final=T_FINAL, mu=MU, gamma=GAMMA):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    t_common = np.linspace(0.0, t_final, 500)
    for idx, (x1, x2) in enumerate(INPUTS):
        sol = simulate(x1, x2, params, n_perc, t_final, mu, gamma)
        X3 = sol.sol(t_common)[0]
        Y = sol.sol(t_common)[1:]
        axes[idx].plot(t_common, X3, label='X3 (output)', linewidth=2)
        for j in range(Y.shape[0]):
            axes[idx].plot(t_common, Y[j], '--', alpha=0.5, label=f'Y{j}' if j == 0 else None)
        axes[idx].axhline(TARGETS[idx], color='r', linestyle=':', label='target')
        axes[idx].set_title(f"X1={x1}, X2={x2}")
        axes[idx].set_xlabel('time')
        axes[idx].set_ylabel('concentration')
        axes[idx].legend()
        axes[idx].grid(True)
    fig.suptitle(f"Full two-timescale RNCRN XOR (M={n_perc}, mu={mu}) {title_suffix}")
    plt.tight_layout()
    fname = f'dack_rncrn_xor_full_M{n_perc}_mu{mu:.0e}_results.png'
    plt.savefig(fname, dpi=150)
    print(f"Saved plot to {fname}")

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    results = {}
    for M in [2, 3, 4]:
        # Note: bounded module uses N_PERC global; we can still use its M=4 params
        # and truncate, but simpler is to train full with M=2,3,4 using warm-start
        # by training bounded for each M. The bounded module has fixed N_PERC=4.
        # For now, just test whether the same 4-perceptron network works when
        # interpreted with fewer active perceptrons. Better: set bounded.N_PERC.
        #
        # We will override N_PERC in the bounded module.
        _bounded.N_PERC = M
        p_opt, best_loss = train_full(M, n_restarts_qs=5, n_restarts_full=3)
        results[M] = (p_opt, best_loss)
        if p_opt is not None:
            preds = predict(p_opt, M)
            print(f"\nFinal XOR truth table for M={M}:")
            for (x1, x2), target, pred in zip(INPUTS, TARGETS, preds):
                print(f"  X1={x1}, X2={x2} -> target={target}, predicted={pred:.4f}")
            plot_results(p_opt, M)

    print("\n=== Summary ===")
    for M, (p_opt, best_loss) in results.items():
        status = "OK" if best_loss < 0.01 else "poor"
        print(f"M={M}: best loss = {best_loss:.6f} ({status})")
