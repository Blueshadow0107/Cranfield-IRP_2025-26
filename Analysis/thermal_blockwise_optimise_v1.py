"""
Blockwise thermal field optimisation.

Treat temperature as a directly tuneable blockwise design variable and use
differential evolution to route a 5 kHz tone to the top probe.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import differential_evolution
from time import perf_counter


# --- domain -------------------------------------------------------------------
NX, NY = 200, 50
LX, LY = 0.400, 0.100
DX = LX / NX
DY = LY / NY

RHO0 = 1.225
T0 = 300.0
C0 = 343.0

FREQ = 5000.0
SRC_AMP = 1.0
SRC_I0, SRC_I1 = 10, 14

SIM_DURATION = 0.006
START_AVG_FRACTION = 0.50

# design region
DESIGN_X0, DESIGN_X1 = 0.160, 0.240
DESIGN_Y0, DESIGN_Y1 = 0.000, LY

# probes
PROBE_A = (170, 37)  # top
PROBE_B = (170, 12)  # bottom

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


# --- blockwise temperature map ------------------------------------------------
def params_to_temperature(params, n_blocks_x, n_blocks_y):
    T = np.full((NX, NY), T0)

    i0 = int(DESIGN_X0 / DX)
    i1 = int(DESIGN_X1 / DX)
    j0 = int(DESIGN_Y0 / DY)
    j1 = int(DESIGN_Y1 / DY)

    bx = (i1 - i0) // n_blocks_x
    by = (j1 - j0) // n_blocks_y

    k = 0
    for iy in range(n_blocks_y):
        y0 = j0 + iy * by
        y1 = y0 + by
        for ix in range(n_blocks_x):
            x0 = i0 + ix * bx
            x1 = x0 + bx
            T[x0:x1, y0:y1] = params[k]
            k += 1

    return T


# --- thermal FDTD --------------------------------------------------------------
def sound_speed(T):
    return C0 * np.sqrt(T / T0)


def run_fdtd(T_field):
    c = sound_speed(T_field)
    c_max = float(c.max())
    dt = 0.9 * DX / (c_max * np.sqrt(2))
    n_steps = int(SIM_DURATION / dt)
    start_avg = int(n_steps * START_AVG_FRACTION)
    omega = 2.0 * np.pi * FREQ

    p = np.zeros((NX, NY))
    u = np.zeros((NX + 1, NY))
    v = np.zeros((NX, NY + 1))

    src_window = np.ones(NY)
    src_window[0] = 0.0
    src_window[-1] = 0.0

    a_history, b_history = [], []

    for n in range(n_steps):
        t = n * dt
        src = SRC_AMP * np.sin(omega * t)
        p[SRC_I0:SRC_I1, :] += 0.25 * src * src_window[None, :]

        u[1:NX, :] -= (dt / RHO0) * (p[1:NX, :] - p[0:NX - 1, :]) / DX
        v[:, 1:NY] -= (dt / RHO0) * (p[:, 1:NY] - p[:, 0:NY - 1]) / DY

        u[0, :] = -p[0, :] / (RHO0 * c[0, :])
        u[NX, :] = p[NX - 1, :] / (RHO0 * c[NX - 1, :])
        v[:, 0] = -p[:, 0] / (RHO0 * c[:, 0])
        v[:, NY] = p[:, NY - 1] / (RHO0 * c[:, NY - 1])

        p -= dt * RHO0 * c**2 * (
            (u[1:NX + 1, :] - u[0:NX, :]) / DX +
            (v[:, 1:NY + 1] - v[:, 0:NY]) / DY
        )

        if n >= start_avg:
            a_history.append(float(p[PROBE_A]))
            b_history.append(float(p[PROBE_B]))

    pa = np.sqrt(np.mean(np.asarray(a_history)**2))
    pb = np.sqrt(np.mean(np.asarray(b_history)**2))
    return pa, pb, p


def fom_from_T(T_field):
    pa, pb, _ = run_fdtd(T_field)
    total = pa**2 + pb**2 + 1e-30
    return (pa**2 / total - 1.0)**2 + (pb**2 / total - 0.0)**2


def objective(params, nbx, nby):
    T = params_to_temperature(params, nbx, nby)
    return fom_from_T(T)


# --- evaluation helper ---------------------------------------------------------
def evaluate(params, nbx, nby, label):
    if nbx == 0 or nby == 0:
        T = np.full((NX, NY), T0)
        fom = fom_from_T(T)
    else:
        T = params_to_temperature(params, nbx, nby)
        fom = objective(params, nbx, nby)
    pa, pb, p = run_fdtd(T)
    total = pa**2 + pb**2
    print(f"  {label}: FOM={fom:.4f}  "
          f"PA={pa:.4e}  PB={pb:.4e}  frac_A={pa**2/total:.3f}")
    return pa, pb, p, T


# --- baseline -----------------------------------------------------------------
print("BASELINE: uniform T = 300 K")
pa0, pb0, p0, T0field = evaluate(np.array([]), 0, 0, "uniform")


# --- optimise -----------------------------------------------------------------
def run_optimisation(nbx, nby, popsize=8, maxiter=15, seed=42):
    n_blocks = nbx * nby
    bounds = [(300.0, 500.0)] * n_blocks
    gen = [0]
    def cb(xk, convergence):
        gen[0] += 1
        print(f"  gen {gen[0]:3d}: best FOM = {convergence:.4f}", flush=True)
    result = differential_evolution(
        objective,
        bounds,
        args=(nbx, nby),
        popsize=popsize,
        maxiter=maxiter,
        seed=seed,
        workers=1,
        polish=False,
        disp=False,
        tol=1e-4,
        atol=1e-4,
        callback=cb
    )
    return result


print("\n" + "=" * 60)
print("OPTIMISATION: 4 x 4 blocks (16 blocks)")
print("=" * 60)
t0 = perf_counter()
res_4x4 = run_optimisation(4, 4, popsize=8, maxiter=15)
elapsed = perf_counter() - t0
print(f"time: {elapsed:.1f} s, best FOM: {res_4x4.fun:.4f}", flush=True)
pa, pb, p, T = evaluate(res_4x4.x, 4, 4, "best 4x4")


# --- figures ------------------------------------------------------------------
def plot_field(T_field, p, title, filename):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))

    ax = axes[0]
    im = ax.imshow(T_field.T, origin='lower', cmap='hot',
                   extent=[0, LX*1000, 0, LY*1000], vmin=T0, vmax=500)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_title('Temperature [K]')
    fig.colorbar(im, ax=ax)

    ax = axes[1]
    vmax = np.abs(p).max()
    im = ax.imshow(p.T.real, origin='lower', cmap='RdBu_r',
                   extent=[0, LX*1000, 0, LY*1000], vmin=-vmax, vmax=vmax)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_title('Re(p) at 5 kHz')
    fig.colorbar(im, ax=ax)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"saved {filename}")


plot_field(T0field, p0, "Baseline uniform temperature", "thermal_blockwise_baseline.png")
plot_field(T, p, f"Best 4x4 design (FOM={res_4x4.fun:.3f})", "thermal_blockwise_4x4.png")



fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogy(res_4x4.funl, label='4x4 blocks')
ax.set_xlabel('function evaluation')
ax.set_ylabel('best FOM')
ax.set_title('Thermal blockwise optimisation convergence')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "thermal_blockwise_convergence.png", dpi=150)
print("saved thermal_blockwise_convergence.png")
