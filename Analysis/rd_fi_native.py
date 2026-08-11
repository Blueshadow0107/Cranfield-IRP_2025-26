"""
STAGE 1 -- Native f-I curve of a sustained-clamp pacemaker port.

Measures the firing-rate-vs-drive-level curve of the Oregonator-A medium
(rd_core.RDSubstrate) when a channel port is held at a SUSTAINED DC level
u = x (drive), v = u* for the whole run.  This is the "native" transfer
curve that Stage 2 (rd_fi_pilot.py) then tries to reshape with tiny
spatial phi parameterisations.

Geometry (per spec):
  - domain 300 x 40, walls everywhere except a straight channel of
    width 16 centred on y = CY
  - port: slab x in [0, 18) spanning the channel width
  - probe: strip at x = 240, 3 cells tall, in-channel (as in
    rd_transfer_channel_final2.strip_mask)

Sustained clamp implementation: the port cells are re-set to (level, u*)
at the top of every step in a custom loop (function run_dc).  This is
exactly equivalent to a rd._holds entry spanning [0, nsteps) -- the same
assignment rd.run() would perform -- but the custom loop also allows the
unclamped-v control (v left free) with an otherwise identical code path.
The port is NOT registered in rd._clamped (no rd.fire call), so the
between-pulse clamp_rest machinery never touches it.

Kinetics: Oregonator Candidate A (f=1.4375, eps=0.050148, q=0.002,
phi0=0.010), rest state u*=v*=0.0030821 via rest_u_star (same algorithm
as rd_transfer_channel_final2.py).  DX=1, DT=0.05 (validated).

Protocol per level: 4000 steps (200 t.u.); first 1000 steps (50 t.u.)
discarded; rate = upward 0.5-crossings (linearly interpolated) of the
probe mean-u in the remainder / 150 t.u.  Also recorded: ISI mean/std
(periodic vs bursting diagnostic), a delay-immune rate estimate
1/mean(ISI), and the counting-noise level 1/sqrt(N).

Wave-speed sanity: the W=16 channel transmits at 6.22 cells/t.u.
(rd_transfer_channel_final2.json), so the 222-cell port->probe transit
takes ~36 t.u. -- comfortably inside the 50 t.u. discard window, i.e.
the crossing-count estimator sees the steady-state train.

Parallelism: levels are independent runs, mapped over a fork-context
pool (NOT forkserver -- see RD/rd_train_xor_phi_fast.py _sim_pool).

Usage:
    ../.venv/bin/python rd_fi_native.py

Outputs: Analysis/figures/rd_fi_native.{png,json}
"""

import json
import multiprocessing
import os
import sys
import time
import warnings

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

# rd_core's domain substep control can divide by a zero rate (harmless:
# the min() picks the finite candidate); silence the noise in worker logs.
warnings.filterwarnings('ignore', category=RuntimeWarning)

# BLAS thread limiting must precede the numpy import so fork workers
# inherit it.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from rd_core import RDSubstrate

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NX, NY = 300, 40
CY = NY // 2
WIDTH = 16                     # channel width (cells)
X_PORT = (0, 18)               # port slab, flush to domain edge
X_PROBE = 240                  # probe strip column
DT = 0.05
NSTEPS = 4000                  # 200 t.u.
DISCARD = 1000                 # first 50 t.u. discarded
U_THRESH = 0.5
Q = 0.002
WORKERS = 8

# Validated excitable Oregonator Candidate A
# (Notes/oregonator-excitable-regime-hunt-2026-07-20.md)
OREG_A = dict(f=1.4375, eps=0.05014844822490394, phi0=0.010)

COARSE_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.65, 0.90]
N_FINE = 10


def rest_u_star(f, phi):
    """Smallest positive root of the u-nullcline with v = u (dv/dt = 0),
    identical to rd_transfer_channel_final2.rest_u_star."""
    def F(u):
        return u - u**2 - (f * u + phi) * (u - Q) / (u + Q)
    ugrid = np.concatenate(([0.0], np.logspace(-7, np.log10(1.2), 4000)))
    vals = F(ugrid)
    for i in range(len(ugrid) - 1):
        if vals[i] > 0.0 and vals[i + 1] <= 0.0:
            return brentq(F, ugrid[i], ugrid[i + 1], xtol=1e-14, rtol=1e-12)
    raise RuntimeError('no rest state found')


U_STAR = rest_u_star(OREG_A['f'], OREG_A['phi0'])


def make_substrate():
    """300x40 domain, walled channel of width 16, port + probe registered."""
    rd = RDSubstrate(nx=NX, ny=NY, dt=DT,
                     kinetics='oregonator', f=OREG_A['f'], eps=OREG_A['eps'],
                     clamp_rest=(U_STAR, U_STAR))
    rd.set_phi(OREG_A['phi0'])
    wall = np.ones((NX, NY), bool)
    wall[:, CY - WIDTH // 2:CY - WIDTH // 2 + WIDTH] = False
    rd.set_walls(wall)
    rd.u[~wall] = U_STAR
    rd.v[~wall] = U_STAR
    port = np.zeros((NX, NY), bool)
    port[X_PORT[0]:X_PORT[1],
         CY - WIDTH // 2:CY - WIDTH // 2 + WIDTH] = True
    rd.add_port('in', port)
    probe = np.zeros((NX, NY), bool)
    probe[X_PROBE, CY - 1:CY + 2] = True
    probe &= ~wall
    rd.add_probe('out', probe)
    return rd


def run_dc(rd, nsteps, level, clamp_v=True):
    """Sustained DC clamp: port cells set to (level, u*) every step.

    With clamp_v=False only u is held; v evolves freely (control for the
    v-clamping physics check).  Returns the probe mean-u series.
    """
    pmask = rd.ports['in']
    qmask = rd.probes['out']
    series = np.empty(nsteps)
    for n in range(nsteps):
        rd.u[pmask] = level
        if clamp_v:
            rd.v[pmask] = U_STAR
        rd._step()
        rd.t += 1
        series[n] = rd.u[qmask].mean()
    return series


def analyse(series):
    """Crossing rate, ISI statistics and counting noise for one trace."""
    seg = series[DISCARD:]
    above = seg >= U_THRESH
    cross = np.nonzero((~above[:-1]) & above[1:])[0]  # upward crossings
    # linearly interpolated crossing times (in t.u., absolute)
    times = []
    for i in cross:
        s0, s1 = seg[i], seg[i + 1]
        frac = (U_THRESH - s0) / (s1 - s0) if s1 > s0 else 0.0
        times.append((DISCARD + i + frac) * DT)
    times = np.array(times)
    n = len(times)
    elapsed = (NSTEPS - DISCARD) * DT
    isi = np.diff(times)
    return {
        'n_crossings': int(n),
        'rate': n / elapsed,                          # per spec
        'rate_isi': float(1.0 / isi.mean()) if n >= 2 else None,
        'isi_mean': float(isi.mean()) if n >= 2 else None,
        'isi_std': float(isi.std()) if n >= 2 else None,
        'isi_cv': float(isi.std() / isi.mean()) if n >= 2 else None,
        'noise_1_over_sqrtN': float(1.0 / np.sqrt(n)) if n > 0 else None,
        'peak': float(seg.max()),
    }


def measure_level(level, clamp_v=True):
    """One pool job: build substrate, run DC clamp, analyse."""
    t0 = time.time()
    rd = make_substrate()
    series = run_dc(rd, NSTEPS, level, clamp_v=clamp_v)
    out = analyse(series)
    out['level'] = float(level)
    out['clamp_v'] = bool(clamp_v)
    out['wall_s'] = time.time() - t0
    return out


def _job(level):
    return measure_level(level)


def control_run():
    """No-stimulus control: the medium must stay at rest."""
    rd = make_substrate()
    rd.run(2000)                       # 100 t.u., no stimulus
    wall = rd.wall
    dev = float(np.abs(rd.u[~wall] - U_STAR).max())
    quiet = dev < 0.01
    print(f'  control (no stimulus, 100 t.u.): max |u-u*| = {dev:.3e} '
          f'-> {"QUIET" if quiet else "SPONTANEOUS FIRING!"}', flush=True)
    return {'rest_u': U_STAR, 'max_dev_100tu': dev, 'quiet': bool(quiet)}


def pool_map(levels):
    ctx = multiprocessing.get_context('fork')
    with ctx.Pool(processes=WORKERS, maxtasksperchild=16) as pool:
        return pool.map(_job, levels, chunksize=1)


def pick_fine_levels(coarse):
    """Concentrate ~N_FINE levels in the silence->saturation ramp region."""
    pts = sorted(((r['level'], r['rate']) for r in coarse))
    xs = np.array([p[0] for p in pts])
    rs = np.array([p[1] for p in pts])
    r_max = rs.max()
    if r_max <= 0.0:
        print('  WARNING: no firing at any coarse level; '
              'falling back to a uniform fine grid', flush=True)
        return list(np.linspace(0.05, 0.9, N_FINE))
    silent = xs[rs <= 0.0]
    x_sil = silent.max() if len(silent) else xs[0]
    sat = xs[rs >= 0.95 * r_max]
    x_sat = sat.min() if len(sat) else xs[-1]
    if x_sat <= x_sil:                 # degenerate: ramp inside one gap
        i = int(np.searchsorted(xs, x_sil))
        x_sat = xs[min(i + 1, len(xs) - 1)]
    lo = max(0.02, x_sil - 0.05)       # a hair into the silent region
    hi = min(0.95, x_sat + 0.05)       # a hair into saturation
    fine = np.linspace(lo, hi, N_FINE)
    print(f'  ramp region: silence up to x~{x_sil:.3f}, '
          f'>=95% of max rate from x~{x_sat:.3f} '
          f'(R_max~{r_max:.4f}/t.u.); fine grid [{lo:.3f}, {hi:.3f}]',
          flush=True)
    return [float(v) for v in fine]


def main():
    t_start = time.time()
    print(f'rest state u* = {U_STAR:.7f} (expect ~0.0030821)', flush=True)
    results = {'config': {
        'nx': NX, 'ny': NY, 'width': WIDTH, 'x_port': list(X_PORT),
        'x_probe': X_PROBE, 'dt': DT, 'nsteps': NSTEPS,
        'discard_steps': DISCARD, 'u_thresh': U_THRESH,
        'oregonator_A': OREG_A, 'u_star': U_STAR,
        'clamp': 'sustained DC: u=level, v=u* every step (custom loop '
                 'equivalent to a full-run _holds entry)',
    }}
    results['control'] = control_run()
    assert results['control']['quiet'], 'control run fires spontaneously!'

    print(f'Coarse scan over {len(COARSE_LEVELS)} levels '
          f'(pool of {WORKERS})...', flush=True)
    coarse = pool_map(COARSE_LEVELS)
    for r in sorted(coarse, key=lambda r: r['level']):
        print(f"  x={r['level']:.3f}: rate={r['rate']:.4f}/t.u. "
              f"(N={r['n_crossings']}, peak={r['peak']:.3f}, "
              f"{r['wall_s']:.0f}s)", flush=True)
    results['coarse'] = sorted(coarse, key=lambda r: r['level'])

    fine_levels = pick_fine_levels(coarse)
    print(f'Fine scan over {len(fine_levels)} levels...', flush=True)
    fine = pool_map(fine_levels)
    for r in sorted(fine, key=lambda r: r['level']):
        isi = (f"ISI={r['isi_mean']:.2f}+-{r['isi_std']:.2f} t.u. "
               f"(CV={r['isi_cv']:.3f})" if r['isi_mean'] is not None
               else 'ISI=n/a')
        print(f"  x={r['level']:.3f}: rate={r['rate']:.4f}/t.u. "
              f"(N={r['n_crossings']}), {isi}", flush=True)
    results['fine'] = sorted(fine, key=lambda r: r['level'])

    # v-clamping physics check at a mid-ramp level (v left free)
    active = [r for r in fine if r['n_crossings'] >= 3]
    if active:
        mid = active[len(active) // 2]['level']
    else:
        mid = float(np.mean(fine_levels))
    print(f'v-clamp check at x={mid:.3f} (v left unclamped)...', flush=True)
    unclamped = measure_level(mid, clamp_v=False)
    clamped = next((r for r in fine if abs(r['level'] - mid) < 1e-12),
                   measure_level(mid, clamp_v=True))
    dv = {'level': mid,
          'rate_clamped_v': clamped['rate'],
          'rate_unclamped_v': unclamped['rate'],
          'isi_mean_clamped_v': clamped['isi_mean'],
          'isi_mean_unclamped_v': unclamped['isi_mean'],
          'rel_rate_diff': (abs(unclamped['rate'] - clamped['rate'])
                            / max(clamped['rate'], 1e-12))}
    print(f"  clamped v: rate={dv['rate_clamped_v']:.4f}/t.u.; "
          f"free v: rate={dv['rate_unclamped_v']:.4f}/t.u. "
          f"(rel diff {100 * dv['rel_rate_diff']:.1f}%)", flush=True)
    results['v_clamp_check'] = dv

    # full level set = coarse + fine merged (used by Stage 2)
    merged = {r['level']: r for r in results['coarse']}
    merged.update({r['level']: r for r in results['fine']})
    results['levels_merged'] = [merged[k] for k in sorted(merged)]
    results['wall_s_total'] = time.time() - t_start

    # ------------------------------------------------------------------
    # Figure: rate curve + ISI panel
    # ------------------------------------------------------------------
    pts = results['levels_merged']
    xs = np.array([r['level'] for r in pts])
    rs = np.array([r['rate'] for r in pts])
    ns = np.array([r['n_crossings'] for r in pts])
    err = np.where(ns > 0, rs / np.sqrt(np.maximum(ns, 1)), 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.errorbar(xs, rs, yerr=err, fmt='o-', ms=4, capsize=3,
                label='native (v clamped)')
    ax.errorbar([dv['level']], [dv['rate_unclamped_v']], fmt='s', ms=7,
                color='tab:red', label='v unclamped (check)')
    ax.set_xlabel('drive level x (port clamp u)')
    ax.set_ylabel('firing rate at probe (1/t.u.)')
    ax.set_title('Native f-I curve (sustained-clamp pacemaker)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ok = [r for r in pts if r['isi_mean'] is not None]
    if ok:
        ax.errorbar([r['level'] for r in ok],
                    [r['isi_mean'] for r in ok],
                    yerr=[r['isi_std'] for r in ok],
                    fmt='o-', ms=4, capsize=3, color='darkgreen')
    ax.set_xlabel('drive level x')
    ax.set_ylabel('inter-spike interval (t.u.)')
    ax.set_title('ISI mean +/- std (periodic vs bursting)')
    ax.grid(alpha=0.3)

    fig.suptitle('Stage 1: native f-I curve, Oregonator A, 300x40 channel '
                 'W=16, probe x=240', fontsize=11)
    fig.tight_layout()
    figpath = 'figures/rd_fi_native.png'
    fig.savefig(figpath, dpi=150)
    plt.close(fig)
    print(f'Saved {figpath}', flush=True)

    jsonpath = 'figures/rd_fi_native.json'
    with open(jsonpath, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f'Saved {jsonpath}', flush=True)
    print(f"Total wall time: {results['wall_s_total'] / 60:.1f} min",
          flush=True)


if __name__ == '__main__':
    main()
