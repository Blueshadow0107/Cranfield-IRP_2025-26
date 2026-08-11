"""
STAGE 2 -- Pilot phi parameterisations for f-I curve shaping.

How much of a target f-I curve shape can tiny spatial phi
parameterisations capture?  Builds on Stage 1 (rd_fi_native.py /
figures/rd_fi_native.json): same geometry (300x40, channel W=16, port
x in [0,18) sustained DC clamp, probe strip x=240), same Oregonator-A
kinetics.

Target: normalised rate y_norm(x) = 0.5*(tanh(8*(x-0.5))+1), evaluated
at a subset of the Stage-1 level set (FIT_LEVELS below, a documented
reduction from the full ~18-level set to keep the compute budget).
R_max is fitted FREELY per candidate (shape-only fit): given measured
rates r_i, R_max = sum(r_i y_i)/sum(y_i^2) (least squares through the
origin), loss = mean((r_i/R_max - y_i)^2).

Designs (phi field over the design region = channel rows, x in
[18, 240); phi = phi0 = 0.010 elsewhere; all phi clipped to the allowed
range [0.010, 0.025]):
  (a) uniform offset: phi = phi0 + d,                     d in [0, 0.015]
  (b) linear ramp:    phi(x) = phi0 + a + b*(x-X0)/L,     a in [0, 0.015],
                                                           b in [-0.015, 0.015]
      (b < 0 with a = 0.015 reaches the valid high-to-low ramp
      0.025 -> 0.010; the clip enforces the phi bounds)
  (c) 2x2 blocks: phi values at the 4 corners of the design rectangle,
      bilinearly interpolated inside; each corner in [0.010, 0.025].

Optimiser: scipy differential_evolution, popsize=6, maxiter=6,
seed=0, vectorized=True, updating='deferred', polish=False.  Candidate
evaluations are flattened into (candidate, level) sim jobs and mapped
over a fork-context pool of 8 (NOT forkserver -- python 3.14 issue, see
RD/rd_train_xor_phi_fast.py _sim_pool).

COMPUTE-BUDGET REDUCTIONS vs Stage 1 (documented; see results json):
  - FIT_LEVELS: 5 levels selected from the Stage-1 merged set to span
    floor / native-vs-target mismatch / ramp / ceiling (fitting all
    ~16 levels x ~294 DE candidates is far over budget).
  - Fit sims are capped at 100 t.u. (vs 200 in Stage 1) with an early
    stop once 12 post-discard crossings are collected; the discard
    window (40 t.u.) exceeds the port->probe transit time (~36 t.u.)
    so the measurement window starts in steady state, and the rate
    estimator is delay-immune ((N-1)/(t_N - t_1)).
  - Per-campaign wall-time budgets halt DE between generations if
    exceeded (documented in the results json when triggered).

Also runs a no-stimulus quiet control for each best-fit phi field.

Usage:
    ../.venv/bin/python rd_fi_pilot.py

Outputs: Analysis/figures/rd_fi_pilot_curves.png,
         Analysis/figures/rd_fi_pilot_results.json
"""

import json
import multiprocessing
import os
import sys
import time
import warnings

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

warnings.filterwarnings('ignore', category=RuntimeWarning)

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq, differential_evolution

from rd_core import RDSubstrate

# ---------------------------------------------------------------------------
# Configuration (geometry identical to Stage 1)
# ---------------------------------------------------------------------------
NX, NY = 300, 40
CY = NY // 2
WIDTH = 16
X_PORT = (0, 18)
X_PROBE = 240
DT = 0.05
U_THRESH = 0.5
Q = 0.002
WORKERS = 8

OREG_A = dict(f=1.4375, eps=0.05014844822490394, phi0=0.010)
PHI0 = OREG_A['phi0']
PHI_MIN, PHI_MAX = 0.010, 0.025

# design region: channel between port and probe
X_D0, X_D1 = X_PORT[1], X_PROBE       # [18, 240)
Y_D0, Y_D1 = CY - WIDTH // 2, CY - WIDTH // 2 + WIDTH

# fit-sim budget (calibrated from Stage 1: native ceiling period
# ~4.2-5 t.u., transit port->probe ~36 t.u. at 6.22 cells/t.u.)
NSTEPS_FIT = 2000            # 100 t.u. cap per sim
DISCARD_FIT = 800            # 40 t.u. >= transit, window starts in steady state
EARLY_STOP_CROSSINGS = 12    # stop once this many post-discard crossings
EARLY_STOP_CHUNK = 100       # check every this many steps
# Stage-1 level subset for the fit (documented reduction from the full
# 16-level merged set): spans floor / native-vs-target mismatch / ramp /
# ceiling of the target tanh centred at x=0.5.
FIT_LEVELS = [0.05, 0.16, 0.45, 0.65, 0.90]
# per-campaign wall-time budgets (min); DE halts between generations
# when exceeded (documented if triggered)
CAMPAIGN_BUDGET_MIN = {'uniform': 20.0, 'ramp': 35.0, 'blocks2x2': 60.0}

NATIVE_JSON = 'figures/rd_fi_native.json'
RESULTS_JSON = 'figures/rd_fi_pilot_results.json'
FIG_PATH = 'figures/rd_fi_pilot_curves.png'


def rest_u_star(f, phi):
    """Identical to rd_transfer_channel_final2.rest_u_star."""
    def F(u):
        return u - u**2 - (f * u + phi) * (u - Q) / (u + Q)
    ugrid = np.concatenate(([0.0], np.logspace(-7, np.log10(1.2), 4000)))
    vals = F(ugrid)
    for i in range(len(ugrid) - 1):
        if vals[i] > 0.0 and vals[i + 1] <= 0.0:
            return brentq(F, ugrid[i], ugrid[i + 1], xtol=1e-14, rtol=1e-12)
    raise RuntimeError('no rest state found')


U_STAR = rest_u_star(OREG_A['f'], PHI0)


def target_ynorm(x):
    """Target normalised rate curve."""
    return 0.5 * (np.tanh(8.0 * (np.asarray(x, dtype=float) - 0.5)) + 1.0)


# ---------------------------------------------------------------------------
# phi field builders
# ---------------------------------------------------------------------------
def build_phi(design, params):
    """Full (nx, ny) phi field for one candidate.  All designs keep
    phi = PHI0 outside the design region; values are clipped to
    [PHI_MIN, PHI_MAX]."""
    field = np.full((NX, NY), PHI0)
    p = np.asarray(params, dtype=float)
    if design == 'uniform':
        region = PHI0 + p[0]
    elif design == 'ramp':
        L = X_D1 - X_D0
        xx = (np.arange(X_D0, X_D1) - X_D0) / L
        region = (PHI0 + p[0] + p[1] * xx)[:, None]   # constant across y
    elif design == 'blocks2x2':
        # p = [phi(x0,y0), phi(x1,y0), phi(x0,y1), phi(x1,y1)] corners,
        # bilinear over the design rectangle
        xx = (np.arange(X_D0, X_D1) - X_D0) / max(X_D1 - X_D0 - 1, 1)
        yy = (np.arange(Y_D0, Y_D1) - Y_D0) / max(Y_D1 - Y_D0 - 1, 1)
        wx = xx[:, None]
        wy = yy[None, :]
        region = (p[0] * (1 - wx) * (1 - wy) + p[1] * wx * (1 - wy)
                  + p[2] * (1 - wx) * wy + p[3] * wx * wy)
    else:
        raise ValueError(design)
    if design == 'uniform':
        field[X_D0:X_D1, Y_D0:Y_D1] = region
    else:
        field[X_D0:X_D1, Y_D0:Y_D1] = np.broadcast_to(
            np.asarray(region), (X_D1 - X_D0, Y_D1 - Y_D0))
    return np.clip(field, PHI_MIN, PHI_MAX)


BOUNDS = {
    'uniform': [(0.0, 0.015)],
    'ramp': [(0.0, 0.015), (-0.015, 0.015)],
    'blocks2x2': [(PHI_MIN, PHI_MAX)] * 4,
}


# ---------------------------------------------------------------------------
# Simulation (same DC-clamp protocol as Stage 1)
# ---------------------------------------------------------------------------
def make_substrate(phi_field):
    rd = RDSubstrate(nx=NX, ny=NY, dt=DT,
                     kinetics='oregonator', f=OREG_A['f'], eps=OREG_A['eps'],
                     clamp_rest=(U_STAR, U_STAR))
    rd.set_phi(phi_field)
    wall = np.ones((NX, NY), bool)
    wall[:, Y_D0:Y_D1] = False
    rd.set_walls(wall)
    rd.u[~wall] = U_STAR
    rd.v[~wall] = U_STAR
    port = np.zeros((NX, NY), bool)
    port[X_PORT[0]:X_PORT[1], Y_D0:Y_D1] = True
    rd.add_port('in', port)
    probe = np.zeros((NX, NY), bool)
    probe[X_PROBE, CY - 1:CY + 2] = True
    probe &= ~wall
    rd.add_probe('out', probe)
    return rd


def run_dc(rd, nsteps, level):
    """Sustained DC clamp (as Stage 1), chunked with an early stop once
    EARLY_STOP_CROSSINGS post-discard crossings are collected (rate
    statistics are then ~1/sqrt(12) ~ 29% at the fastest firing, while
    slow/suppressed candidates simply run to the step cap)."""
    pmask = rd.ports['in']
    qmask = rd.probes['out']
    chunks = []
    n_done = 0
    while n_done < nsteps:
        m = min(EARLY_STOP_CHUNK, nsteps - n_done)
        seg = np.empty(m)
        for n in range(m):
            rd.u[pmask] = level
            rd.v[pmask] = U_STAR
            rd._step()
            rd.t += 1
            seg[n] = rd.u[qmask].mean()
        chunks.append(seg)
        n_done += m
        if n_done > DISCARD_FIT:
            above = np.concatenate(chunks)[DISCARD_FIT:] >= U_THRESH
            if int(np.sum((~above[:-1]) & above[1:])) \
                    >= EARLY_STOP_CROSSINGS:
                break
    return np.concatenate(chunks)


def rate_from_series(series):
    """Delay-immune rate: (N-1)/(t_N - t_1) over post-discard upward
    0.5-crossing times (falls back to N/elapsed for N == 1).  Unlike the
    raw crossings/elapsed estimator this is insensitive to exactly where
    the discard window ends relative to the wave train."""
    seg = series[DISCARD_FIT:]
    above = seg >= U_THRESH
    cross = np.nonzero((~above[:-1]) & above[1:])[0]
    n = len(cross)
    elapsed = len(seg) * DT
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0 / elapsed
    return (n - 1) / ((cross[-1] - cross[0]) * DT)


# ---------------------------------------------------------------------------
# Parallel evaluation: flattened (candidate, level) jobs over a fork pool
# ---------------------------------------------------------------------------
_SIM_POOL = None


def _sim_pool():
    """Fork-context pool of WORKERS=8 (NOT forkserver; see
    RD/rd_train_xor_phi_fast.py for the python 3.14 story)."""
    global _SIM_POOL
    if _SIM_POOL is None:
        ctx = multiprocessing.get_context('fork')
        _SIM_POOL = ctx.Pool(processes=WORKERS, maxtasksperchild=64)
    return _SIM_POOL


def _sim_job(job):
    """One (design, params, level) sim; returns (key, level, rate)."""
    key, design, params, level = job
    try:
        rd = make_substrate(build_phi(design, params))
        series = run_dc(rd, NSTEPS_FIT, level)
        return (key, level, rate_from_series(series))
    except Exception:
        return (key, level, None)


class FitProblem:
    """Objective for one design over the shared fit level set."""

    def __init__(self, design, levels, y_target):
        self.design = design
        self.levels = list(levels)
        self.y = np.asarray(y_target, dtype=float)
        self.n_evals = 0

    def objective_vectorized(self, X):
        """X: (dim, S) candidate matrix -> losses (S,)."""
        S = X.shape[1]
        jobs = []
        for j in range(S):
            params = np.ascontiguousarray(X[:, j])
            for lv in self.levels:
                jobs.append((j, self.design, params, lv))
        t0 = time.time()
        outs = _sim_pool().map(_sim_job, jobs, chunksize=1)
        rates = {j: {} for j in range(S)}
        for key, level, rate in outs:
            rates[key][level] = rate
        losses = np.empty(S)
        for j in range(S):
            r = np.array([rates[j][lv] if rates[j][lv] is not None
                          else 0.0 for lv in self.levels])
            losses[j] = shape_loss(r, self.y)[0]
        self.n_evals += S
        print(f'[de:{self.design}] generation batch of {S} candidates '
              f'done in {(time.time() - t0) / 60:.1f} min '
              f'(best loss {losses.min():.5f})', flush=True)
        return losses


def shape_loss(rates, y):
    """Shape-only fit: R_max through the origin, MSE of normalised rates."""
    r = np.asarray(rates, dtype=float)
    denom = float(np.sum(y**2))
    r_max = float(np.sum(r * y) / denom) if denom > 0 else 0.0
    if r_max <= 0.0:
        return float(np.mean(y**2)), 0.0, np.zeros_like(r)
    r_norm = r / r_max
    return float(np.mean((r_norm - y)**2)), r_max, r_norm


# ---------------------------------------------------------------------------
# Level-set selection and main driver
# ---------------------------------------------------------------------------
class BudgetCallback:
    """Halt a DE campaign between generations once its wall-time budget
    is spent (scipy honours a True return by stopping)."""

    def __init__(self, budget_min, tag):
        self.budget_s = budget_min * 60.0
        self.tag = tag
        self.t0 = time.time()
        self.stopped = False

    def __call__(self, xk, convergence=0.0):
        if time.time() - self.t0 > self.budget_s:
            self.stopped = True
            print(f'[de:{self.tag}] WALL-TIME BUDGET reached; halting '
                  f'early (partial result)', flush=True)
            return True
        return False


def quiet_control(phi_field, nsteps=2000):
    """No-stimulus control with a shaped phi field: must stay at rest."""
    rd = make_substrate(phi_field)
    rd.run(nsteps)
    dev = float(np.abs(rd.u[~rd.wall] - U_STAR).max())
    return {'max_dev': dev, 'quiet': bool(dev < 0.01)}


def main():
    t_start = time.time()
    with open(NATIVE_JSON) as fh:
        native = json.load(fh)
    merged = native['levels_merged']

    levels = list(FIT_LEVELS)
    y = target_ynorm(levels)
    print(f'Fit levels ({len(levels)}): {[round(v, 3) for v in levels]}',
          flush=True)
    print(f'Target y_norm: {[round(float(v), 3) for v in y]}', flush=True)

    # native baseline shape error at the same levels (rates from Stage 1)
    native_rates = np.array([next(r['rate'] for r in merged
                                  if abs(r['level'] - lv) < 1e-12)
                             for lv in levels])
    nat_loss, nat_rmax, nat_norm = shape_loss(native_rates, y)
    print(f'Native shape loss vs target: {nat_loss:.5f} '
          f'(R_max={nat_rmax:.4f}/t.u.)', flush=True)

    results = {'config': {
        'levels': levels, 'target_y': [float(v) for v in y],
        'nsteps_fit': NSTEPS_FIT, 'discard_fit': DISCARD_FIT,
        'phi_bounds': [PHI_MIN, PHI_MAX],
        'design_region': {'x': [X_D0, X_D1], 'y': [Y_D0, Y_D1]},
        'de': {'popsize': 6, 'maxiter': 6, 'seed': 0, 'polish': False},
        'early_stop_crossings': EARLY_STOP_CROSSINGS,
        'campaign_budget_min': CAMPAIGN_BUDGET_MIN,
        'reductions': 'fit on a 5-level subset of the Stage-1 level set '
                      '(floor/mismatch/ramp/ceiling), sims capped at 100 '
                      't.u. with an early stop after 12 post-discard '
                      'crossings, and per-campaign wall-time budgets, to '
                      'keep total compute under ~2 h',
    }, 'native': {'rates': [float(v) for v in native_rates],
                  'r_max': nat_rmax, 'shape_loss': nat_loss,
                  'rates_norm': [float(v) for v in nat_norm]},
        'designs': {}}

    fits = {}
    for design in ('uniform', 'ramp', 'blocks2x2'):
        print(f'=== DE campaign: {design} ===', flush=True)
        prob = FitProblem(design, levels, y)
        t0 = time.time()
        budget = BudgetCallback(CAMPAIGN_BUDGET_MIN[design], design)
        res = differential_evolution(
            prob.objective_vectorized, bounds=BOUNDS[design],
            strategy='best1bin', popsize=6, maxiter=6, tol=1e-4,
            mutation=(0.5, 1.0), recombination=0.7, seed=0,
            vectorized=True, workers=1, updating='deferred',
            polish=False, disp=True, callback=budget)
        wall = time.time() - t0
        best = np.asarray(res.x, dtype=float)
        # re-evaluate the best candidate's rates for the record
        outs = _sim_pool().map(
            _sim_job,
            [(0, design, best, lv) for lv in levels], chunksize=1)
        r_best = np.array([o[2] if o[2] is not None else 0.0
                           for o in outs])
        loss, r_max, r_norm = shape_loss(r_best, y)
        phi_field = build_phi(design, best)
        ctrl = quiet_control(phi_field)
        entry = {'best_params': [float(v) for v in best],
                 'de_loss': float(res.fun), 'nfev': int(res.nfev),
                 'budget_stopped_early': bool(budget.stopped),
                 'reeval_loss': loss, 'r_max': r_max,
                 'rates': [float(v) for v in r_best],
                 'rates_norm': [float(v) for v in r_norm],
                 'shape_error_captured_vs_native':
                     (1.0 - loss / nat_loss) if nat_loss > 0 else None,
                 'control': ctrl, 'wall_s': wall}
        results['designs'][design] = entry
        fits[design] = entry
        print(f'  {design}: loss={loss:.5f} (native {nat_loss:.5f}), '
              f'R_max={r_max:.4f}, params={[round(float(v),5) for v in best]}, '
              f'control quiet={ctrl["quiet"]} (dev {ctrl["max_dev"]:.2e}), '
              f'{wall/60:.1f} min', flush=True)

    results['wall_s_total'] = time.time() - t_start

    # ------------------------------------------------------------------
    # Figure: native vs target vs the three best fits
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(levels, y, 'k--', lw=1.5, label='target tanh')
    ax.plot(levels, nat_norm, 'o-', ms=5, color='gray',
            label=f'native (loss {nat_loss:.4f})')
    colors = {'uniform': 'tab:blue', 'ramp': 'tab:orange',
              'blocks2x2': 'tab:green'}
    for design, entry in fits.items():
        ax.plot(levels, entry['rates_norm'], 's-', ms=4,
                color=colors[design],
                label=f"{design} (loss {entry['reeval_loss']:.4f})")
    ax.set_xlabel('drive level x')
    ax.set_ylabel('normalised firing rate')
    ax.set_title('Stage 2 pilot: f-I curve shaping via tiny phi fields\n'
                 '(Oregonator A, design region x in [18,240), '
                 'phi in [0.010, 0.025])', fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)
    print(f'Saved {FIG_PATH}', flush=True)

    with open(RESULTS_JSON, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f'Saved {RESULTS_JSON}', flush=True)
    print(f"Total wall time: {results['wall_s_total'] / 60:.1f} min",
          flush=True)


if __name__ == '__main__':
    main()
