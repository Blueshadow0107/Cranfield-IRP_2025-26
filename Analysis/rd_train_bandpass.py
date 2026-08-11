"""
rd_train_bandpass.py -- train a phi(x) profile in a straight BZ channel to act
as a tunable low-pass / frequency-discrimination gate.

Task:
    Input A: slow pulse train (period 5.0 t.u., 4 pulses)  -> output >= 3 peaks.
    Input B: fast pulse train (period 2.0 t.u., 6 pulses)  -> output <= 1 peak.

The idea is to shape the effective refractory dynamics along the channel so
that the natural low-pass filtering is enhanced: slow trains propagate, fast
trains are attenuated by dropping pulses.  This is a non-trivial use of
refractory dynamics, not reducible to a single blocking barrier.

Optimiser: CMA-ES (Covariance Matrix Adaptation Evolution Strategy).

Outputs:
    Analysis/figures/rd_train_bandpass/
        result.json         -- best parameters, loss, metrics
        phi_profile.png     -- best phi(x) profile
        traces.png          -- probe time-series for A and B
        convergence.png     -- CMA-ES fitness history
"""
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import cma

from rd_core import RDSubstrate

# -----------------------------------------------------------------------------
# Physics and working point
# -----------------------------------------------------------------------------
F = 1.4375
EPS = 0.05014844822490394
Q = 0.002
PHI0 = 0.010          # background excitable level
DARK = 0.002          # dark-spot phi during flash
DT = 0.05
U_STAR = 0.0030821    # homogeneous rest state at phi0
U_THRESH = 0.5

# -----------------------------------------------------------------------------
# Domain and geometry
# -----------------------------------------------------------------------------
NX, NY = 200, 48
CY = NY // 2
WIDTH = 16
X_PORT = (0, 18)
X_PROBE = 170
NSTEPS = 800          # enough for pulse to reach probe

# Channel walls: open strip in y
WALL = np.ones((NX, NY), bool)
j0 = CY - WIDTH // 2
j1 = j0 + WIDTH
WALL[:, j0:j1] = False

# Input port (left end of channel)
PORT = np.zeros((NX, NY), bool)
PORT[X_PORT[0]:X_PORT[1], j0:j1] = True

# Output probe (far end, small disk)
PROBE_R = 5
X, Y = np.mgrid[0:NX, 0:NY]
PROBE = ((X - X_PROBE)**2 + (Y - CY)**2 <= PROBE_R**2) & ~WALL

# -----------------------------------------------------------------------------
# Trainable phi profile
# -----------------------------------------------------------------------------
N_CTRL = 8          # number of control points along x
PHI_MIN, PHI_MAX = 0.010, 0.024   # keep inside excitable window
X_CTRL = np.linspace(30, 150, N_CTRL)  # control region excludes port/probe


def build_phi(ctrl):
    """1-D control values -> full (NX, NY) phi field, uniform across y."""
    ctrl = np.asarray(ctrl, dtype=float)
    # pad with background phi outside control region
    x_all = np.concatenate(([0.0], X_CTRL, [NX - 1.0]))
    v_all = np.concatenate(([PHI0], ctrl, [PHI0]))
    f = interp1d(x_all, v_all, kind='linear', fill_value=PHI0, bounds_error=False)
    phi_x = f(np.arange(NX))
    phi_x = np.clip(phi_x, PHI_MIN, PHI_MAX)
    phi = np.tile(phi_x[:, None], (1, NY))
    phi[WALL] = PHI0
    return phi


# -----------------------------------------------------------------------------
# Forward solve for a given input pattern
# -----------------------------------------------------------------------------
def run_case(phi, pulse_times, flash_duration_steps, label='',
             save_full=False, stride=5):
    """Run one simulation. pulse_times are step indices at which flashes start.

    If save_full is True, also returns full u(x,y,t), v(x,y,t) and phi(x,y)
    snapshots sampled every `stride` steps.  This is used for post-training
    analysis and visualisation; CMA-ES evaluations use only the probe trace.
    """
    rd = RDSubstrate(nx=NX, ny=NY, dt=DT, kinetics='oregonator',
                     f=F, eps=EPS, clamp_rest=(U_STAR, U_STAR))
    rd.set_walls(WALL)
    rd.set_phi(phi)
    rd.u[~WALL] = U_STAR
    rd.v[~WALL] = U_STAR
    rd.add_probe('out', PROBE)

    u_probe = []
    if save_full:
        u_snaps, v_snaps, t_snaps = [], [], []

    # Run with dynamic phi: dark during flashes, background otherwise
    for s in range(NSTEPS):
        in_flash = any(t <= s < t + flash_duration_steps for t in pulse_times)
        if in_flash:
            phi_flash = phi.copy()
            phi_flash[PORT] = DARK
            rd.set_phi(phi_flash)
        else:
            rd.set_phi(phi)
        rd.run(1)
        u_probe.append(rd.u[PROBE].max() if PROBE.any() else 0.0)
        if save_full and (s % stride == 0 or s == NSTEPS - 1):
            u_snaps.append(rd.u.copy())
            v_snaps.append(rd.v.copy())
            t_snaps.append(s * DT)

    u_probe = np.asarray(u_probe)
    t = np.arange(len(u_probe)) * DT
    if save_full:
        return (t, u_probe,
                np.asarray(t_snaps),
                np.stack(u_snaps, axis=0),
                np.stack(v_snaps, axis=0),
                phi.copy())
    return t, u_probe


# -----------------------------------------------------------------------------
# Metrics and loss
# -----------------------------------------------------------------------------
def count_peaks(series, thresh=U_THRESH):
    above = series >= thresh
    return int(np.sum((~above[:-1]) & above[1:]))


def probe_metrics(series):
    peak = float(series.max())
    n_peaks = count_peaks(series)
    return {'peak': peak, 'n_peaks': n_peaks}


def evaluate(ctrl, verbose=False):
    phi = build_phi(ctrl)

    # Case A: slow train (period 5.0 t.u., 4 pulses) -> target 3+ output peaks
    period_slow = int(5.0 / DT)
    dur_steps = int(1.5 / DT)
    times_slow = [k * period_slow for k in range(4)]
    tA, uA = run_case(phi, times_slow, dur_steps, 'slow')
    mA = probe_metrics(uA)

    # Case B: fast train (period 2.0 t.u., 6 pulses) -> target <= 1 output peak
    period_fast = int(2.0 / DT)
    times_fast = [k * period_fast for k in range(6)]
    tB, uB = run_case(phi, times_fast, dur_steps, 'fast')
    mB = probe_metrics(uB)

    # Loss: slow train should pass most pulses; fast train should be strongly attenuated
    loss_A = max(0.0, 3 - mA['n_peaks'])          # penalise if slow train produces fewer than 3 peaks
    loss_B = max(0.0, mB['n_peaks'] - 1)          # penalise if fast train produces more than 1 peak
    loss = loss_A + loss_B

    # small TV penalty on phi slope (prefer smooth profiles)
    phi_x = build_phi(ctrl)[:, CY]
    tv = float(np.abs(np.diff(phi_x)).mean())
    loss += 0.01 * tv

    if verbose:
        print(f"  slow {mA['n_peaks']}/4 peaks  fast {mB['n_peaks']}/6 peaks  "
              f"loss={loss:.3f} (A={loss_A:.3f} B={loss_B:.3f} tv={tv:.3f})")

    return loss, {'loss': loss, 'loss_A': loss_A, 'loss_B': loss_B, 'tv': tv,
                  'ctrl': ctrl.tolist(), 'mA': mA, 'mB': mB}


# -----------------------------------------------------------------------------
# CMA-ES training
# -----------------------------------------------------------------------------
def _eval_worker(x):
    """Top-level worker for multiprocessing; returns (loss, rec)."""
    return evaluate(np.asarray(x))


def train(maxfevals=200, popsize=None, n_workers=8):
    outdir = Path(__file__).parent / 'figures' / 'rd_train_bandpass'
    outdir.mkdir(parents=True, exist_ok=True)

    x0 = np.full(N_CTRL, PHI0)
    sigma0 = 0.003
    bounds = [PHI_MIN, PHI_MAX]

    opts = {
        'bounds': bounds,
        'maxfevals': maxfevals,
        'verbose': -9,  # minimal CMA output
        'verb_log': 0,
    }
    if popsize is not None:
        opts['popsize'] = popsize

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    history = []
    best = (1e9, None, None)
    with Pool(processes=n_workers) as pool:
        while not es.stop():
            solutions = es.ask()
            results = pool.map(_eval_worker, solutions, chunksize=1)
            fitnesses = []
            for loss, rec in results:
                fitnesses.append(loss)
                history.append({'feval': len(history) + 1, 'loss': loss,
                                'mA_peak': rec['mA']['peak'],
                                'mB_peak': rec['mB']['peak']})
                if loss < best[0]:
                    best = (loss, rec['ctrl'].copy(), rec)
            es.tell(solutions, fitnesses)
            print(f"[CMA] gen done, nfev={len(history)}, best={best[0]:.4f}")

    print(f"[CMA] best loss={best[0]:.4f}  nfev={len(history)}")

    # Final evaluation with verbose output
    loss, rec = evaluate(np.asarray(best[1]), verbose=True)

    # Save results
    result = {
        'best_ctrl': best[1],
        'best_loss': best[0],
        'nfev': len(history),
        'history': history,
        'metrics': {'A': rec['mA'], 'B': rec['mB']},
    }
    with open(outdir / 'result.json', 'w') as fh:
        json.dump(result, fh, indent=2)

    # Figures (probe traces + convergence)
    plot_phi(best[1], outdir / 'phi_profile.png')
    plot_traces(best[1], outdir / 'traces.png')
    plot_convergence(history, outdir / 'convergence.png')

    # Full-field snapshots for the best candidate
    print('[snap] recording full u/v/phi fields for best candidate...')
    save_snapshots(np.asarray(best[1]), outdir / 'snapshots.npz')
    plot_spacetime(best[1], outdir / 'spacetime.png')

    print(f"[saved] {outdir}")
    return np.asarray(best[1]), best[0]


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_phi(ctrl, path):
    phi = build_phi(ctrl)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(np.arange(NX), phi[:, CY], 'k-', lw=2, label=r'$\phi(x)$')
    ax.axhspan(PHI_MIN, PHI_MAX, alpha=0.1, color='gray')
    ax.axvline(X_PROBE, color='g', ls='--', label='probe')
    ax.set_xlabel('x (cells)')
    ax.set_ylabel(r'$\phi$')
    ax.set_title('Best trained phi profile')
    ax.set_ylim(PHI_MIN - 0.002, PHI_MAX + 0.002)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_traces(ctrl, path):
    phi = build_phi(ctrl)

    dur_steps = int(1.5 / DT)
    tA, uA = run_case(phi, [k * int(5.0 / DT) for k in range(4)], dur_steps, 'slow')
    tB, uB = run_case(phi, [k * int(2.0 / DT) for k in range(6)], dur_steps, 'fast')

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    axes[0].plot(tA, uA, 'b-', lw=1.5)
    axes[0].axhline(U_THRESH, color='r', ls='--', label='threshold')
    axes[0].set_ylabel('u at probe')
    axes[0].set_title('Slow train: 4 in, target >= 3 out peaks')
    axes[0].legend()

    axes[1].plot(tB, uB, 'g-', lw=1.5)
    axes[1].axhline(U_THRESH, color='r', ls='--', label='threshold')
    axes[1].set_xlabel('time (t.u.)')
    axes[1].set_ylabel('u at probe')
    axes[1].set_title('Fast train: 6 in, target <= 1 out peak')
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_convergence(history, path):
    losses = [h['loss'] for h in history]
    best_so_far = np.minimum.accumulate(losses)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(losses) + 1), losses, 'k.', alpha=0.3, label='per-eval')
    ax.plot(range(1, len(best_so_far) + 1), best_so_far, 'r-', lw=2, label='best so far')
    ax.set_xlabel('function evaluation')
    ax.set_ylabel('loss')
    ax.set_title('CMA-ES convergence')
    ax.legend()
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Full-field snapshots and space--time visualisation
# -----------------------------------------------------------------------------
def save_snapshots(ctrl, path, stride=5):
    """Save full u/v/phi snapshots for the slow and fast train cases."""
    phi = build_phi(ctrl)
    dur_steps = int(1.5 / DT)

    res_slow = run_case(phi, [k * int(5.0 / DT) for k in range(4)],
                        dur_steps, 'slow', save_full=True, stride=stride)
    res_fast = run_case(phi, [k * int(2.0 / DT) for k in range(6)],
                        dur_steps, 'fast', save_full=True, stride=stride)

    np.savez(path,
             phi=phi,
             t_slow=res_slow[2], u_slow=res_slow[3], v_slow=res_slow[4],
             t_fast=res_fast[2], u_fast=res_fast[3], v_fast=res_fast[4],
             ctrl=ctrl, x=np.arange(NX), y=np.arange(NY))
    print(f'[saved] {path}')


def plot_spacetime(ctrl, path):
    """Space--time strip of u at channel centre for slow and fast trains."""
    phi = build_phi(ctrl)
    dur_steps = int(1.5 / DT)

    _, _, t_slow, u_slow, _, _ = run_case(
        phi, [k * int(5.0 / DT) for k in range(4)],
        dur_steps, 'slow', save_full=True, stride=5)
    _, _, t_fast, u_fast, _, _ = run_case(
        phi, [k * int(2.0 / DT) for k in range(6)],
        dur_steps, 'fast', save_full=True, stride=5)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for ax, t, u, title in zip(
            axes, [t_slow, t_fast], [u_slow, u_fast],
            ['Slow train (4 in, target >= 3 out)',
             'Fast train (6 in, target <= 1 out)']):
        # u at centreline, masked by walls
        u_strip = u[:, :, CY].copy()
        u_strip[:, WALL[:, CY]] = np.nan
        im = ax.imshow(u_strip.T, aspect='auto', origin='lower',
                       extent=[t[0], t[-1], 0, NX - 1],
                       cmap='viridis', vmin=0.0, vmax=0.8)
        ax.set_ylabel('x (cells)')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='u')
    axes[1].set_xlabel('time (t.u.)')
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    maxfevals = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    popsize = int(sys.argv[2]) if len(sys.argv) > 2 else None
    n_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    train(maxfevals=maxfevals, popsize=popsize, n_workers=n_workers)
