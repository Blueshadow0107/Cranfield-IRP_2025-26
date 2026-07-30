"""Train a spatial light-suppression field phi(x,y) so the Test-3 T-junction
computes XOR -- the first training campaign of the RD-computing project.

Headline experiment (2026-07-20).  Hand-design could not make the junction
compute XOR: the native geometry computes OR (or, with a narrow window on
A's arrival only, A AND NOT B -- see rd_transfer_logic_final2.py).  Here the
64 values of an 8x8 block grid of phi over a design region around the
junction are trained by differential evolution.

Setup (geometry, kinetics, readout inherited from rd_transfer_logic_final2.py,
Oregonator branch -- do NOT drift from it):
  - 256x200 grid, dx=1, dt=0.05, channel width W=20, junction x TJ=96.
  - A channel: horizontal, y in [90,110).  B channel: vertical, x in
    [86,106), y in [100,172).  Output probe strip x=240, y in [99,102).
  - Port A: x[25,43) x channel; port B: channel x y[140,158); pulses hold
    u=0.8, v=0.2 for 30 steps.
  - Oregonator Candidate A: f=1.4375, eps=0.050148448..., q=0.002,
    baseline illumination phi0=0.010, rest u*=v*=0.0030821.
  - Medium initialised at the UNIFORM-phi0 rest state; the phi pattern is a
    perturbation (projector over background illumination).

Readout protocol (FROZEN before training -- computed once from uniform-phi0
reference runs and cached in figures/rd_train_xor_protocol.json):
  - tA, tB = first u>=0.5 crossing at the probe for A-alone / B-alone.
  - Window = [min(tA,tB)-2.5, max(tA,tB)+2.5] t.u.  (final2 used a +/-2.5
    t.u. half-window around tA; XOR needs BOTH lone inputs readable at the
    same probe, so the window is extended symmetrically to cover B's later
    arrival.  With this window the uniform-phi0 baseline reads OR:
    p10~p01~p11~A0, separation ~1 -- the number to beat.)
  - Training sims stop 2.5 t.u. after the window closes (the readout only
    sees the window; nothing later matters).  Robustness re-runs use the
    full 1000-step traces.

Weight space and loss:
  - phi trained on an 8x8 block grid over the design region
    x in [44,224], y in [64,176] (covers the junction plus the A-channel
    segment downstream of port A, the whole B channel, and the output
    channel up to just before the probe column); bilinear interpolation
    onto the fine grid; phi clipped to [0.010, 0.025] (the pattern can only
    ADD light over background; >=0.03 kills propagation globally).
  - loss = mean[(p10/A0-1)^2, (p01/A0-1)^2, (p00/A0)^2, (p11/A0)^2]
           + LAMBDA_TV * mean(|grad(block phi)|),  A0 = lone-pulse window
    peak of the uniform-phi0 reference (~0.7).
  - LAMBDA_TV calibration: mean|grad| of a random 8x8 draw in
    [0.010,0.025] is ~5e-3, so LAMBDA_TV=1.0 contributes ~5e-3 to the loss
    -- ~1% of the O(0.5) data term; it regularises speckle without ever
    outweighing the truth-table mismatch.  Verified against the smoke-test
    eval log.
  - If the rd_core guard trips (FloatingPointError) or any other numerical
    failure occurs, the candidate is logged as failed with loss FAIL_LOSS
    and the campaign continues.

Optimiser: scipy.optimize.differential_evolution, workers=-1 (16-way
parallel candidate evaluation; the 4 cases of a truth table run
sequentially inside each worker), polish=False, popsize/maxiter set from
the measured pilot cost so the campaign fits the ~10 h budget, seed fixed.

Subcommands:
    protocol  run the 4 uniform-phi0 reference sims, freeze + cache the
              window/A0/baseline truth table
    smoke     2-generation, popsize-4 DE run to validate the pipeline
    train     full DE campaign (budget from CLI flags)
    robust    robustness evaluation of the best trained design:
              full-resolution truth table + separation, timing-jitter test
              (dB = -20, 0, +20 steps), phi-map figure
    demux     FALLBACK: fresh DE campaign for the demultiplexer task if XOR
              separation stays < 3x (see DEMUX note at the bottom)

Outputs (all new files):
    figures/rd_train_xor_protocol.json     frozen protocol + baseline
    figures/rd_train_xor_eval_log.jsonl    per-candidate eval log
    figures/rd_train_xor_checkpoint.json   DE checkpoint (best x so far)
    figures/rd_train_xor_history.png       loss vs eval
    figures/rd_train_xor_phi_map.png       best design
    figures/rd_train_xor_truth_tables.png  before/after
    figures/rd_train_xor_jitter.png        jitter degradation
    figures/rd_train_xor_results.json      all metrics

Usage:  ../.venv/bin/python rd_train_xor_phi.py <subcommand> [options]
"""

import argparse
import json
import multiprocessing
import os
import sys
import time
import warnings
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# rd_core's domain substep control can divide by a zero rate (harmless:
# the min() picks the finite candidate); the FloatingPointError guard
# still catches genuine blow-ups.  Silence the noise in worker logs.
warnings.filterwarnings('ignore', category=RuntimeWarning)

# BLAS thread limiting must precede the numpy import so the forkserver
# workers (which re-import this module) inherit it.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq, differential_evolution
from scipy.interpolate import RegularGridInterpolator
from contextlib import redirect_stdout, redirect_stderr

from rd_core import RDSubstrate
import logging

# Load configuration
with open('config.json') as f:
    config = json.load(f)

# ---------------------------------------------------------------------------
# Frozen physics / geometry (rd_transfer_logic_final2.py, Oregonator branch)
# ---------------------------------------------------------------------------
Q = 0.002
DT = 0.05
W = 20
W2 = W // 2
TJ = 96                    # junction x
X_PROBE = 240
NX, NY = 256, 200
CY = NY // 2
B_CH_TOP = 172
F = 1.4375
EPS = 0.05014844822490394
PHI0 = config['parameters']['phi0']
PHI_MAX = config['parameters']['phi_max']
DURATION = 30
U_THRESH = 0.5
WIN_HALF = 2.5             # t.u. half-window (final2 convention)
NSTEP_FULL = 1000          # full trace length (final2 convention)
TRAIL = 2.5                # t.u. margin after window close for training sims

# ---------------------------------------------------------------------------
# Trainable field: 8x8 block grid over the design region
# ---------------------------------------------------------------------------
NB = config['parameters']['nb_blocks']
REGION_X = tuple(config['parameters']['region_x'])
REGION_Y = tuple(config['parameters']['region_y'])
LAMBDA_TV = config['parameters']['lambda_tv']
FAIL_LOSS = config['parameters']['fail_loss']
SEED = config['parameters']['seed']

FIG = config['paths']['figures']
PROTOCOL_JSON = config['paths']['protocol']
CHECKPOINT_JSON = config['paths']['checkpoint']
RESULTS_JSON = config['paths']['results']

CASES = (('00', False, False), ('10', True, False),
         ('01', False, True), ('11', True, True))


# ---------------------------------------------------------------------------
# Rest state and geometry (verbatim from rd_transfer_logic_final2.py)
# ---------------------------------------------------------------------------
def rest_u_star(f, phi):
    def Fn(u):
        return u - u**2 - (f * u + phi) * (u - Q) / (u + Q)
    g = np.concatenate(([0.0], np.logspace(-7, np.log10(1.2), 4000)))
    v = Fn(g)
    for i in range(len(g) - 1):
        if v[i] > 0 and v[i + 1] <= 0:
            return brentq(Fn, g[i], g[i + 1], xtol=1e-14, rtol=1e-12)
    raise RuntimeError('no rest state found')


US = rest_u_star(F, PHI0)  # 0.0030821


def make_rd(phi):
    """Build the Test-3 T-junction with light field `phi` (scalar or
    (nx, ny)).  Mirrors rd_transfer_logic_final2.make_rd('oregonator')."""
    wall = np.ones((NX, NY), bool)
    wall[:, CY - W2:CY + W2] = False                  # A channel -> output
    wall[TJ - W2:TJ + W2, CY:B_CH_TOP] = False        # B channel from top
    rd = RDSubstrate(nx=NX, ny=NY, dt=DT, kinetics='oregonator',
                     f=F, eps=EPS, clamp_rest=(US, US))
    rd.set_phi(phi)
    rd.set_walls(wall)
    rd.u[~wall] = US
    rd.v[~wall] = US
    pA = np.zeros((NX, NY), bool)
    pA[25:43, CY - W2:CY + W2] = True
    pB = np.zeros((NX, NY), bool)
    pB[TJ - W2:TJ + W2, 140:158] = True
    rd.add_port('A', pA)
    rd.add_port('B', pB)
    probe = np.zeros((NX, NY), bool)
    probe[X_PROBE, CY - 1:CY + 2] = True
    rd.add_probe('out', probe & ~wall)
    return rd, wall


def run_case(phi, fire_a, fire_b, dB=0, nsteps=NSTEP_FULL, tA_fire=0):
    logging.info(f'[run_case] starting sim: A={fire_a}, B={fire_b}')
    """Run one input case.  A fires at step tA_fire, B at tA_fire+dB
    (dB>=0).  For dB<0 use tA_fire=-dB so both pulses keep the full
    30-step hold (relative timing identical to A@-0, B@dB)."""
    rd, _ = make_rd(phi)
    if fire_a:
        rd.pulse_train('A', [tA_fire], duration=DURATION)
    if fire_b:
        rd.pulse_train('B', [tA_fire + dB], duration=DURATION)
    return rd.run(nsteps)


# ---------------------------------------------------------------------------
# Block-grid phi field
# ---------------------------------------------------------------------------
def build_phi(blocks):
    """Map 64 block values (8x8, row-major over x then y) to the (nx, ny)
    field: bilinear interpolation of block centres across the design
    region, phi0 outside, clipped to [PHI0, PHI_MAX]."""
    V = np.asarray(blocks, dtype=float).reshape(NB, NB)
    bx = (REGION_X[1] - REGION_X[0]) / NB
    by = (REGION_Y[1] - REGION_Y[0]) / NB
    xc = REGION_X[0] + (np.arange(NB) + 0.5) * bx   # block-centre coords
    yc = REGION_Y[0] + (np.arange(NB) + 0.5) * by
    interp = RegularGridInterpolator((xc, yc), V, method='linear',
                                     bounds_error=False, fill_value=None)
    i0, i1 = int(REGION_X[0]), int(REGION_X[1])
    j0, j1 = int(REGION_Y[0]), int(REGION_Y[1])
    xi = (np.arange(i0, i1) + 0.5)
    yj = (np.arange(j0, j1) + 0.5)
    pts = np.stack(np.meshgrid(xi, yj, indexing='ij'), axis=-1)
    field = np.full((NX, NY), PHI0)
    field[i0:i1, j0:j1] = interp(pts)
    return np.clip(field, PHI0, PHI_MAX)


def tv_term(blocks):
    """Mean |grad| of the 8x8 block array (total-variation regulariser)."""
    V = np.asarray(blocks).reshape(NB, NB)
    gx, gy = np.gradient(V)
    return float(0.5 * (np.abs(gx).mean() + np.abs(gy).mean()))


# ---------------------------------------------------------------------------
# Frozen readout protocol
# ---------------------------------------------------------------------------
def first_crossing(t, s, thresh=U_THRESH):
    idx = np.nonzero(s >= thresh)[0]
    return float(t[idx[0]]) if len(idx) else None


def window_peak(series, win_lo, win_hi, probe='out'):
    m = (series['t'] >= win_lo) & (series['t'] <= win_hi)
    if not m.any():
        return 0.0
    return float(series[probe][m].max())


def compute_protocol():
    """Run the 4 uniform-phi0 reference sims and freeze the readout:
    arrival times tA/tB, the window covering both, A0, the baseline
    (windowed) truth table, and the training sim length."""
    logging.info('[protocol] running 4 uniform-phi0 reference sims '
          f'(u*={US:.7f}) ...')
    t0 = time.time()
    ref = {}
    for tag, fa, fb in CASES:
        ref[tag] = run_case(PHI0, fa, fb)
        logging.info(f'[protocol]   case {tag} done '
              f'({time.time() - t0:.0f}s elapsed)')
    tA = first_crossing(ref['10']['t'], ref['10']['out'])
    tB = first_crossing(ref['01']['t'], ref['01']['out'])
    if tA is None or tB is None:
        raise RuntimeError('baseline lone-input wave did not reach probe')
    win_lo = min(tA, tB) - WIN_HALF
    win_hi = max(tA, tB) + WIN_HALF
    base = {tag: window_peak(ref[tag], win_lo, win_hi) for tag, _, _ in CASES}
    A0 = 0.5 * (base['10'] + base['01'])
    nsteps_train = int(np.ceil((win_hi + TRAIL) / DT))
    sep = min(base['10'], base['01']) / max(base['00'], base['11'], 1e-3)
    proto = {
        'kinetics': 'Oregonator Candidate A (f=1.4375, eps=0.0501, q=0.002)',
        'phi0': PHI0, 'phi_max': PHI_MAX, 'rest_u_star': US,
        'geometry': ('256x200, W=20, TJ=96, A port x[25,43), B port '
                     'y[140,158), B channel y<172, probe strip x=240'),
        'tA_arrival': tA, 'tB_arrival': tB,
        'window_tu': [win_lo, win_hi],
        'window_rule': ('[min(tA,tB)-2.5, max(tA,tB)+2.5] t.u.; final2 '
                        'half-window extended to cover both lone-input '
                        'arrivals so XOR true cases are both readable'),
        'A0_lone_peak': A0,
        'nsteps_train': nsteps_train,
        'nsteps_full': NSTEP_FULL,
        'design_region': {'x': REGION_X, 'y': REGION_Y, 'blocks': NB},
        'lambda_tv': LAMBDA_TV,
        'baseline_truth_windowed': base,
        'baseline_separation_xor': sep,
        'baseline_note': ('windowed baseline reads OR: p10~p01~p11~A0, '
                          'separation ~1'),
        'protocol_wall_s': time.time() - t0,
    }
    with open(PROTOCOL_JSON, 'w') as fh:
        json.dump(proto, fh, indent=2)
    logging.info(f'[protocol] tA={tA:.2f}, tB={tB:.2f}, window='
          f'[{win_lo:.2f},{win_hi:.2f}] t.u., A0={A0:.3f}, '
          f'nsteps_train={nsteps_train}, baseline sep={sep:.2f}x')
    logging.info(f'[protocol] saved {PROTOCOL_JSON}')
    return proto


def load_protocol(recompute=False):
    if not recompute and os.path.exists(PROTOCOL_JSON):
        with open(PROTOCOL_JSON) as fh:
            return json.load(fh)
    return compute_protocol()


# ---------------------------------------------------------------------------
# Objective (runs in DE worker processes)
# ---------------------------------------------------------------------------
def truth_table_peaks(phi, nsteps, win_lo, win_hi):
    """Windowed probe peaks for the four input cases."""
    peaks = {}
    for tag, fa, fb in CASES:
        data = run_case(phi, fa, fb, nsteps=nsteps)
        peaks[tag] = window_peak(data, win_lo, win_hi)
    return peaks


def objective(x, proto):
    """DE objective: XOR loss + small TV regulariser.  Never raises on
    numerical failure -- failed candidates get FAIL_LOSS."""
    win_lo, win_hi = proto['window_tu']
    A0 = proto['A0_lone_peak']
    rec = {'t': time.time(), 'failed': False}
    try:
        phi = build_phi(x)
        pk = truth_table_peaks(phi, proto['nsteps_train'], win_lo, win_hi)
        terms = [((pk['10'] / A0) - 1.0) ** 2,
                 ((pk['01'] / A0) - 1.0) ** 2,
                 (pk['00'] / A0) ** 2,
                 (pk['11'] / A0) ** 2]
        data_term = float(np.mean(terms))
        tv = tv_term(x)
        loss = data_term + LAMBDA_TV * tv
        rec.update(peaks=pk, data_term=data_term, tv=tv, loss=float(loss))
    except FloatingPointError as exc:
        loss = FAIL_LOSS
        rec.update(failed=True, error=f'FloatingPointError: {exc}',
                   loss=FAIL_LOSS)
    except Exception as exc:  # keep the campaign alive on any failure
        loss = FAIL_LOSS
        rec.update(failed=True, error=f'{type(exc).__name__}: {exc}',
                   loss=FAIL_LOSS)
    try:
        with open(EVAL_LOG, 'a') as fh:
            fh.write(json.dumps(rec) + '\n')
    except OSError as e:
        logging.error(f"Failed to write to eval log: {e}")
    return float(loss)


class CampaignLogger:
    """DE callback: checkpoint the incumbent after every generation."""

    def __init__(self, tag):
        self.tag = tag
        self.gen = 0
        self.t0 = time.time()

    def __call__(self, xk, convergence):
        print(f'[CampaignLogger] callback called for generation {self.gen + 1}', flush=True)
        self.gen += 1
        state = {'tag': self.tag, 'generation': self.gen,
                 'best_x': list(map(float, xk)),
                 'elapsed_s': time.time() - self.t0,
                 'convergence': float(convergence)}
        prev = {}
        if os.path.exists(CHECKPOINT_JSON):
            try:
                with open(CHECKPOINT_JSON) as fh:
                    prev = json.load(fh)
            except (OSError, json.JSONDecodeError):
                prev = {}
        hist = prev.get('gen_history', [])
        hist.append({'gen': self.gen, 'convergence': float(convergence)})
        state['gen_history'] = hist
        with open(CHECKPOINT_JSON, 'w') as fh:
            json.dump(state, fh)
        print(f'[de:{self.tag}] generation {self.gen} done, '
              f'convergence={convergence:.4f}, '
              f'elapsed={state["elapsed_s"] / 60:.1f} min', flush=True)


# ---------------------------------------------------------------------------
# Flat-parallel objective (population candidates over a fork-context pool)
# ---------------------------------------------------------------------------
_SIM_POOL = None


def _sim_pool():
    """Lazily created pool of 12 candidate workers.

    Uses the FORK start method, NOT forkserver: on Python 3.14 the POSIX
    default changed to forkserver, whose workers re-import __main__ and
    pay a heavy startup tax (measured: scipy workers=4 gave only ~2x on
    a 2s-sleep micro-test, and the production run effectively
    serialised).  Fork children inherit this module's state directly.
    maxtasksperchild recycles workers periodically to avoid resource
    leaks over a long campaign."""
    global _SIM_POOL
    if _SIM_POOL is None:
        ctx = multiprocessing.get_context('fork')
        _SIM_POOL = ctx.Pool(processes=12, maxtasksperchild=64)
    return _SIM_POOL


def _run_case_safe(job):
    """One (phi, case) job; returns the run dict, or None on failure."""
    phi, fa, fb, nsteps = job
    try:
        return run_case(phi, fa, fb, nsteps=nsteps)
    except Exception:
        return None


def objective_vectorized(X, proto):
    """Vectorized DE objective: X has shape (dim, S); returns losses (S,).

    Each candidate is one pool task running the plain serial objective
    (4 sims + eval-log row).  12 candidates in flight gives ~12-way core
    utilisation with per-candidate streaming to the eval log."""
    S = X.shape[1]
    print(f'[obj_vec] evaluating {S} candidates over pool', flush=True)
    xs = [np.ascontiguousarray(X[:, j]) for j in range(S)]
    t0 = time.time()
    losses = _sim_pool().starmap(
        objective, [(x, proto) for x in xs], chunksize=1)
    print(f'[obj_vec] generation batch done in '
          f'{(time.time() - t0) / 60:.1f} min', flush=True)
    return np.asarray(losses, dtype=float)


def run_de(proto, popsize, maxiter, tag):
    """One DE campaign over the 64 block values."""
    logging.info(f'[run_de] starting campaign: {tag}')
    logger = CampaignLogger(tag)
    t0 = time.time()
    result = differential_evolution(
        objective_vectorized, bounds=[(PHI0, PHI_MAX)] * (NB * NB),
        args=(proto,),
        strategy='best1bin', popsize=popsize, maxiter=maxiter,
        tol=1e-4, mutation=(0.5, 1.0), recombination=0.7,
        seed=SEED, vectorized=True, workers=1, updating='deferred',
        polish=False,
        x0=np.full(NB * NB, PHI0),
        callback=logger, disp=True)
    out = {'tag': tag, 'popsize': popsize, 'maxiter': maxiter,
           'seed': SEED, 'nfev': int(result.nfev),
           'best_loss': float(result.fun),
           'best_x': [float(v) for v in result.x],
           'wall_s': time.time() - t0, 'message': str(result.message)}
    return out


# ---------------------------------------------------------------------------
# Evaluation-log helpers and figures
# ---------------------------------------------------------------------------
def read_eval_log():
    recs = []
    if os.path.exists(EVAL_LOG):
        with open(EVAL_LOG) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        recs.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    recs.sort(key=lambda r: r['t'])
    return recs


def fig_history(path):
    recs = read_eval_log()
    if not recs:
        logging.info('[fig] no eval log yet -- skipping history figure')
        return
    losses = np.array([r['loss'] for r in recs])
    best = np.minimum.accumulate(losses)
    nfail = int(sum(r.get('failed', False) for r in recs))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(1, len(losses) + 1), losses, '.', ms=2, alpha=0.4,
            label='candidate loss')
    ax.plot(np.arange(1, len(losses) + 1), best, '-', lw=1.5,
            label='best so far')
    ax.set_yscale('log')
    ax.set_xlabel('truth-table evaluation #')
    ax.set_ylabel('loss')
    ax.set_title(f'XOR phi-training: loss vs eval ({len(losses)} evals, '
                 f'{nfail} failed)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logging.info(f'[fig] saved {path}')


def fig_phi_map(blocks, path, title):
    field = build_phi(blocks)
    wall = np.ones((NX, NY), bool)
    wall[:, CY - W2:CY + W2] = False
    wall[TJ - W2:TJ + W2, CY:B_CH_TOP] = False
    fig, ax = plt.subplots(figsize=(9, 6))
    shown = np.ma.masked_where(wall, field)
    im = ax.imshow(shown.T, origin='lower', cmap='viridis',
                   vmin=PHI0, vmax=PHI_MAX)
    ax.contour(wall.T, levels=[0.5], colors='white', linewidths=0.6)
    rx = [REGION_X[0], REGION_X[1], REGION_X[1], REGION_X[0], REGION_X[0]]
    ry = [REGION_Y[0], REGION_Y[0], REGION_Y[1], REGION_Y[1], REGION_Y[0]]
    ax.plot(rx, ry, 'r--', lw=1.0, label='design region')
    bx = (REGION_X[1] - REGION_X[0]) / NB
    by = (REGION_Y[1] - REGION_Y[0]) / NB
    for k in range(1, NB):
        ax.axvline(REGION_X[0] + k * bx, color='r', lw=0.2, alpha=0.5)
        ax.axhline(REGION_Y[0] + k * by, color='r', lw=0.2, alpha=0.5)
    ax.plot(X_PROBE, CY, 'w^', ms=8, label='probe')
    ax.set_xlabel('x (cells)')
    ax.set_ylabel('y (cells)')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)
    fig.colorbar(im, ax=ax, label='phi')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logging.info(f'[fig] saved {path}')


def fig_truth_tables(base, trained, path, thr):
    labels = ['00', '10', '01', '11']
    targets = [0, 1, 1, 0]
    xpos = np.arange(4)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(xpos - 0.2, [base[k] for k in labels], width=0.4,
           label='baseline (uniform phi0) = OR', color='gray')
    ax.bar(xpos + 0.2, [trained[k] for k in labels], width=0.4,
           label='trained phi', color='tab:blue')
    for i, tgt in enumerate(targets):
        ax.text(i, -0.06 * max(1.0, max(trained.values())),
                f'target {tgt}', ha='center', fontsize=9)
    ax.axhline(thr, color='k', ls='--', lw=0.8,
               label=f'decision thr {thr:.2f}')
    ax.set_xticks(xpos)
    ax.set_xticklabels([f'({k[0]},{k[1]})' for k in labels])
    ax.set_xlabel('inputs (A,B)')
    ax.set_ylabel('windowed peak u at output probe')
    ax.set_title('XOR truth table: baseline OR vs trained phi')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logging.info(f'[fig] saved {path}')


def fig_jitter(jitter, path, thr):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for tag, marker in (('10', 'o'), ('01', 's'), ('11', '^')):
        ds = [j['dB'] for j in jitter[tag]]
        ps = [j['peak'] for j in jitter[tag]]
        ax.plot(ds, ps, marker + '-', label=f'({tag[0]},{tag[1]})')
    ax.axhline(thr, color='k', ls='--', lw=0.8,
               label=f'decision thr {thr:.2f}')
    ax.set_xlabel('B firing delay dB (steps)')
    ax.set_ylabel('windowed peak u at output probe')
    ax.set_title('Trained XOR: timing-jitter degradation')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logging.info(f'[fig] saved {path}')


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------
def cmd_protocol(args):
    load_protocol(recompute=True)


def cmd_smoke(args):
    proto = load_protocol()
    logging.info('[smoke] 2 generations, popsize 4 (pipeline validation)')
    out = run_de(proto, popsize=4, maxiter=2, tag='smoke')
    logging.info(f'[smoke] done: nfev={out["nfev"]}, best_loss={out["best_loss"]:.4f}, '
          f'wall={out["wall_s"] / 60:.1f} min')
    logging.info(f'[smoke] per-eval cost ~{out["wall_s"] / out["nfev"]:.2f} s '
          f'(16-way parallel)')
    fig_history(os.path.join(FIG, 'rd_train_xor_history.png'))


def cmd_train(args):
    proto = load_protocol()
    out = run_de(proto, popsize=args.popsize, maxiter=args.maxiter,
                 tag='train')
    logging.info(f'[train] done: nfev={out["nfev"]}, '
          f'best_loss={out["best_loss"]:.4f}, '
          f'wall={out["wall_s"] / 3600:.2f} h')
    with open(CHECKPOINT_JSON, 'w') as fh:
        json.dump({'tag': 'train-final', 'best_x': out['best_x'],
                   'best_loss': out['best_loss'], 'de': out}, fh, indent=2)
    fig_history(os.path.join(FIG, 'rd_train_xor_history.png'))


def cmd_robust(args):
    """Robustness evaluation of the best design in the checkpoint."""
    proto = load_protocol()
    with open(CHECKPOINT_JSON) as fh:
        ckpt = json.load(fh)
    best_x = np.array(ckpt['best_x'], dtype=float)
    win_lo, win_hi = proto['window_tu']
    base = proto['baseline_truth_windowed']

    # (1) full-resolution truth table of the trained design
    logging.info('[robust] full-resolution truth table of best design ...')
    phi = build_phi(best_x)
    t0 = time.time()
    trained = {}
    for tag, fa, fb in CASES:
        trained[tag] = window_peak(run_case(phi, fa, fb), win_lo, win_hi)
        logging.info(f'[robust]   ({tag[0]},{tag[1]}): {trained[tag]:.4f} '
              f'({time.time() - t0:.0f}s)')
    sep_tr = min(trained['10'], trained['01']) / \
        max(trained['00'], trained['11'], 1e-3)
    sep_ba = proto['baseline_separation_xor']
    thr = 0.5 * (min(trained['10'], trained['01'])
                 + max(trained['00'], trained['11']))
    logging.info(f'[robust] separation: baseline {sep_ba:.2f}x -> '
          f'trained {sep_tr:.2f}x')

    # (2) timing jitter: dB in {-20, 0, +20} steps
    logging.info('[robust] timing-jitter test ...')
    jitter = {}
    for tag, fa, fb in (('10', True, False), ('01', False, True),
                        ('11', True, True)):
        rows = []
        for dB in (-20, 0, 20):
            # dB<0: fire B at 0 and A at -dB (full pulses preserved);
            # window shifts with A's firing time.
            tA_fire = max(0, -dB)
            shift = tA_fire * DT
            ns = NSTEP_FULL + tA_fire
            data = run_case(phi, fa, fb, dB=max(dB, 0),
                            nsteps=ns, tA_fire=tA_fire)
            pk = window_peak(data, win_lo + shift, win_hi + shift)
            rows.append({'dB': dB, 'peak': pk})
            logging.info(f'[robust]   ({tag[0]},{tag[1]}) dB={dB:+d}: '
                  f'peak={pk:.4f}')
        jitter[tag] = rows

    # (3) figures + results json
    fig_phi_map(best_x, os.path.join(FIG, 'rd_train_xor_phi_map.png'),
                f'Trained phi field (loss={ckpt.get("best_loss", float("nan")):.4f})')
    fig_truth_tables(base, trained,
                     os.path.join(FIG, 'rd_train_xor_truth_tables.png'), thr)
    fig_jitter(jitter, os.path.join(FIG, 'rd_train_xor_jitter.png'), thr)
    fig_history(os.path.join(FIG, 'rd_train_xor_history.png'))

    results = {
        'experiment': 'train phi(x,y) for XOR at the Test-3 T-junction',
        'date': '2026-07-20',
        'protocol': proto,
        'de': ckpt.get('de', {'tag': ckpt.get('tag')}),
        'baseline_truth_windowed': base,
        'baseline_separation_xor': sep_ba,
        'trained_truth_windowed': trained,
        'trained_separation_xor': sep_tr,
        'separation_target': '>=3x (else fallback to demultiplexer)',
        'fallback_needed': bool(sep_tr < 3.0),
        'decision_threshold': thr,
        'jitter': jitter,
        'best_blocks': [float(v) for v in best_x],
        'robust_wall_s': time.time() - t0,
    }
    with open(RESULTS_JSON, 'w') as fh:
        json.dump(results, fh, indent=2)
    logging.info(f'[robust] saved {RESULTS_JSON}')
    if sep_tr < 3.0:
        logging.info('[robust] SEPARATION < 3x -- XOR-not-found; '
              'fallback demultiplexer campaign indicated '
              '(subcommand demux).')
    else:
        logging.info(f'[robust] XOR FOUND: separation {sep_tr:.2f}x >= 3x')


# ---------------------------------------------------------------------------
# FALLBACK: demultiplexer (only used if XOR separation < 3x)
# ---------------------------------------------------------------------------
# Geometry variant (documented choice): cross junction.  A enters from the
# left in the horizontal channel as before; B is the control input from the
# top; the vertical channel is extended DOWNWARD from the junction to
# y=28 so there are two output ports: P1 = straight through (x=240, y=CY)
# and P2 = deflected downward (x=TJ, y=30).  Task: A alone -> P1 (p1 high,
# p2 low); A+B -> P2 (p1 low, p2 high).  2 sims per truth table (B never
# fires alone -- B is a pure control), so a campaign of the same eval count
# costs ~half of the XOR one.  Loss:
#   mean[(p1(B=0)/A0-1)^2, (p2(B=0)/A0)^2,
#        (p1(B=1)/A0)^2,       (p2(B=1)/A0-1)^2] + LAMBDA_TV*tv
DEMUX_P2_Y = 30
DEMUX_CH_BOT = 28


def make_rd_demux(phi):
    wall = np.ones((NX, NY), bool)
    wall[:, CY - W2:CY + W2] = False                  # A -> P1 channel
    wall[TJ - W2:TJ + W2, DEMUX_CH_BOT:B_CH_TOP] = False  # vertical through
    rd = RDSubstrate(nx=NX, ny=NY, dt=DT, kinetics='oregonator',
                     f=F, eps=EPS, clamp_rest=(US, US))
    rd.set_phi(phi)
    rd.set_walls(wall)
    rd.u[~wall] = US
    rd.v[~wall] = US
    pA = np.zeros((NX, NY), bool)
    pA[25:43, CY - W2:CY + W2] = True
    pB = np.zeros((NX, NY), bool)
    pB[TJ - W2:TJ + W2, 140:158] = True
    rd.add_port('A', pA)
    rd.add_port('B', pB)
    p1 = np.zeros((NX, NY), bool)
    p1[X_PROBE, CY - 1:CY + 2] = True
    rd.add_probe('P1', p1 & ~wall)
    p2 = np.zeros((NX, NY), bool)
    p2[TJ - 1:TJ + 2, DEMUX_P2_Y] = True
    rd.add_probe('P2', p2 & ~wall)
    return rd, wall


def cmd_demux(args):
    """Fallback demultiplexer campaign: A routed to P1 vs P2 by control B.

    Protocol frozen here: window covers the A-alone arrivals at P1 and P2
    is read in a window around A+B's expected P2 arrival -- both measured
    from uniform-phi0 reference runs (documented in the demux results).
    """
    proto = load_protocol()
    logging.info('[demux] reference runs (uniform phi0) ...')
    ref = {}
    for tag, fb in (('B0', False), ('B1', True)):
        rd, _ = make_rd_demux(PHI0)
        rd.pulse_train('A', [0], duration=DURATION)
        if fb:
            rd.pulse_train('B', [0], duration=DURATION)
        ref[tag] = rd.run(NSTEP_FULL)
    tP1 = first_crossing(ref['B0']['t'], ref['B0']['P1'])
    tP2 = first_crossing(ref['B1']['t'], ref['B1']['P2'])
    win1 = (tP1 - WIN_HALF, tP1 + WIN_HALF)
    win2 = ((tP2 - WIN_HALF, tP2 + WIN_HALF) if tP2 is not None else win1)
    p10 = window_peak(ref['B0'], *win1, probe='P1')
    A0 = p10
    nsteps_train = int(np.ceil((max(win1[1], win2[1]) + TRAIL) / DT))
    demux_proto = {'win1': win1, 'win2': win2, 'A0': A0,
                   'nsteps_train': nsteps_train,
                   'tP1': tP1, 'tP2': tP2,
                   'task': ('A alone -> P1, A+B -> P2 (control B routes '
                            'input A); cross-junction variant of Test-3')}
    logging.info(f'[demux] tP1={tP1}, tP2={tP2}, A0={A0:.3f}, '
          f'nsteps_train={nsteps_train}')

    def demux_obj(x):
        rec = {'t': time.time(), 'failed': False, 'task': 'demux'}
        try:
            phi = build_phi(x)
            peaks = {}
            for tag, fb in (('B0', False), ('B1', True)):
                rd, _ = make_rd_demux(phi)
                rd.pulse_train('A', [0], duration=DURATION)
                if fb:
                    rd.pulse_train('B', [0], duration=DURATION)
                d = rd.run(nsteps_train)
                peaks[tag] = (window_peak(d, *win1, probe='P1'),
                              window_peak(d, *win2, probe='P2'))
            (p1_0, p2_0), (p1_1, p2_1) = peaks['B0'], peaks['B1']
            terms = [((p1_0 / A0) - 1.0) ** 2, (p2_0 / A0) ** 2,
                     (p1_1 / A0) ** 2, ((p2_1 / A0) - 1.0) ** 2]
            data_term = float(np.mean(terms))
            loss = data_term + LAMBDA_TV * tv_term(x)
            rec.update(peaks={'B0': peaks['B0'], 'B1': peaks['B1']},
                       data_term=data_term, loss=float(loss))
        except Exception as exc:
            loss = FAIL_LOSS
            rec.update(failed=True, error=f'{type(exc).__name__}: {exc}',
                       loss=FAIL_LOSS)
        try:
            with open(EVAL_LOG, 'a') as fh:
                fh.write(json.dumps(rec) + '\n')
        except OSError:
            pass
        return float(loss)

    logger = CampaignLogger('demux')
    t0 = time.time()
    result = differential_evolution(
        demux_obj, bounds=[(PHI0, PHI_MAX)] * (NB * NB),
        strategy='best1bin', popsize=args.popsize, maxiter=args.maxiter,
        tol=1e-4, mutation=(0.5, 1.0), recombination=0.7,
        seed=SEED + 1, workers=-1, updating='deferred', polish=False,
        x0=np.full(NB * NB, PHI0), callback=logger, disp=False)
    out = {'task': demux_proto['task'], 'protocol': demux_proto,
           'popsize': args.popsize, 'maxiter': args.maxiter,
           'seed': SEED + 1, 'nfev': int(result.nfev),
           'best_loss': float(result.fun),
           'best_x': [float(v) for v in result.x],
           'wall_s': time.time() - t0, 'message': str(result.message)}
    path = os.path.join(FIG, 'rd_train_demux_results.json')
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=2)
    logging.info(f'[demux] done: nfev={out["nfev"]}, '
          f'best_loss={out["best_loss"]:.4f}, '
          f'wall={out["wall_s"] / 3600:.2f} h; saved {path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('protocol', help='freeze the readout protocol')
    sub.add_parser('smoke', help='2-gen popsize-4 pipeline validation')
    for name in ('train', 'demux'):
        p = sub.add_parser(name)
        p.add_argument('--popsize', type=int, default=8)
        p.add_argument('--maxiter', type=int, default=8)
    sub.add_parser('robust', help='robustness eval of best design')
    args = ap.parse_args()

    # --- Dynamic Log File Setup ---
    start_time = time.strftime("%Y%m%d-%H%M%S")
    
    # Set up logging to a dynamic file
    log_filename = f'debug_{start_time}.log'
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=log_filename,
                        filemode='w')
    
    # Make EVAL_LOG a global variable so it can be accessed in objective()
    global EVAL_LOG
    EVAL_LOG = os.path.join(config['paths']['figures'], f'rd_train_xor_eval_log_{start_time}.jsonl')

    logging.info(f"Starting run with log file: {log_filename}")
    logging.info(f"Evaluation log will be at: {EVAL_LOG}")
    
    {'protocol': cmd_protocol, 'smoke': cmd_smoke, 'train': cmd_train,
     'robust': cmd_robust, 'demux': cmd_demux}[args.cmd](args)


if __name__ == '__main__':
    main()
