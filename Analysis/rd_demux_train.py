"""
rd_demux_train.py -- routing/demultiplexer training campaign (token-free).

Train phi(x,y) so one-shot dark-spot patterns route to assigned probes:
    (10) -> PA1, (01) -> PA5, (11) -> PA3   (window [2, 8] t.u. after flash)

Geometry: V5 (user-locked). 200x200 free medium, spots A(40,80) B(40,120)
r=12, one-shot dark flash phi=0.002 for 12 t.u., probes PC(95,100),
PA1(82,26), PA2(114,58), PA3(125,100), PA4(114,142), PA5(82,174).
Design region x[55,135] y[20,180], 8x8 blocks bilinear, phi in [0.010,
0.040], >=10-cell probe halos excluded.  (Run 1 used PHI_MAX=0.025 and
was stopped after ~490 flat evals: rd_phi_barrier.py showed blocking
starts only between 0.025 and 0.030, so the run-1 band could slow but
never stop a pulse.  0.040 adds real blocking headroom.)

Loss (ungameable): presence (peak>=0.5 in window) + wrong-channel count +
normalised timing error, per pattern; + small TV penalty.  Attenuation
cannot help because presence is required.

DE: popsize 4 (x64 dims = 256/gen), maxiter 4, best1bin, seed fixed,
vectorized objective over a FORK-context pool of 12 (py3.14 forkserver
is broken; fork inherits module state).

Outputs (Analysis/figures/): rd_demux_protocol.json,
rd_demux_eval_log.jsonl (one line per eval, flushed live),
rd_demux_status.json (live progress: evals done, best loss, ETA),
rd_demux_checkpoint.json, rd_demux_results.json,
rd_demux_history.png, rd_demux_phi_map.png, rd_demux_traces.png

Watch live:  tail -f Analysis/figures/rd_demux_eval_log.jsonl
             cat  Analysis/figures/rd_demux_status.json
"""

import json
import multiprocessing
import os
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import differential_evolution

from rd_core import RDSubstrate

# ---------------------------------------------------------------------------
# Geometry / physics (locked V5)
# ---------------------------------------------------------------------------
NX, NY = 200, 200
A_POS, B_POS = (40, 80), (40, 120)
SPOT_R = 12
PROBES = {'PC': (95, 100), 'PA1': (82, 26), 'PA2': (114, 58),
          'PA3': (125, 100), 'PA4': (114, 142), 'PA5': (82, 174)}
F, EPS, PHI0 = 1.4375, 0.05014844822490394, 0.010
U_STAR = 0.0030821
DARK = 0.002
DT = 0.05
T_FLASH = 12.0
WINDOW = (2.0, 8.0)          # t.u. after flash end
TU_AFTER = 60.0
U_THRESH = 0.5

DR_X, DR_Y = (55, 135), (20, 180)
NB = 8
PHI_MIN, PHI_MAX = 0.010, 0.040
HALO = 10
LAMBDA_TV = 0.02
FAIL_LOSS = 10.0
SEED = 20260802
POOL_WORKERS = 12
MAXITER = 4
POP = 4 * NB * NB                    # DE population per generation
EVALS_TOTAL = POP * (MAXITER + 1)    # initial pop + MAXITER generations
LOG_EVERY = 8                        # stdout progress line cadence

ASSIGN = {'10': 'PA1', '01': 'PA5', '11': 'PA3'}
FIRE = {'10': (True, False), '01': (False, True), '11': (True, True)}
PATTERNS = list(ASSIGN.keys())

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG, exist_ok=True)


def spot_mask(pos):
    X, Y = np.mgrid[0:NX, 0:NY].astype(float)
    return ((X - pos[0]) ** 2 + (Y - pos[1]) ** 2) < SPOT_R ** 2


HALO_MASK = np.zeros((NX, NY), bool)
for _p in PROBES.values():
    _X, _Y = np.mgrid[0:NX, 0:NY].astype(float)
    HALO_MASK |= ((_X - _p[0]) ** 2 + (_Y - _p[1]) ** 2) < HALO ** 2


def make_rd(phi):
    rd = RDSubstrate(nx=NX, ny=NY, dt=DT, kinetics='oregonator',
                     f=F, eps=EPS, clamp_rest=(U_STAR, U_STAR))
    rd.set_phi(phi)
    rd.u[:] = U_STAR
    rd.v[:] = U_STAR
    return rd


def build_phi(blocks):
    V = np.asarray(blocks, dtype=float).reshape(NB, NB)
    bx = (DR_X[1] - DR_X[0]) / NB
    by = (DR_Y[1] - DR_Y[0]) / NB
    xc = DR_X[0] + (np.arange(NB) + 0.5) * bx
    yc = DR_Y[0] + (np.arange(NB) + 0.5) * by
    interp = RegularGridInterpolator((xc, yc), V, method='linear',
                                     bounds_error=False, fill_value=None)
    xi = np.arange(DR_X[0], DR_X[1]) + 0.5
    yj = np.arange(DR_Y[0], DR_Y[1]) + 0.5
    pts = np.stack(np.meshgrid(xi, yj, indexing='ij'), axis=-1)
    field = np.full((NX, NY), PHI0)
    field[DR_X[0]:DR_X[1], DR_Y[0]:DR_Y[1]] = interp(pts)
    field[HALO_MASK] = PHI0
    return np.clip(field, PHI_MIN, PHI_MAX)


def run_pattern(phi, fire_a, fire_b):
    """One pattern: flash the spot(s) dark for T_FLASH inside the candidate
    field, restore the candidate field, record all probes for TU_AFTER."""
    rd = make_rd(phi)
    phi_dark = phi.copy()
    if fire_a:
        phi_dark[spot_mask(A_POS)] = DARK
    if fire_b:
        phi_dark[spot_mask(B_POS)] = DARK
    rd.set_phi(phi_dark)
    rd.run(int(T_FLASH / DT))
    rd.set_phi(phi)
    series = {name: np.empty(int(TU_AFTER / DT)) for name in PROBES}
    for n in range(int(TU_AFTER / DT)):
        rd.run(1)
        for name, pos in PROBES.items():
            series[name][n] = rd.u[pos]
    return series


def probe_metrics(series):
    """Per probe: first arrival (t.u.) and in-window peak."""
    out = {}
    t = np.arange(series.shape[0]) * DT
    idx = np.nonzero(series > U_THRESH)[0]
    first = float(t[idx[0]]) if len(idx) else None
    m = (t >= WINDOW[0]) & (t <= WINDOW[1])
    out['first'] = first
    out['peak'] = float(series[m].max()) if m.any() else 0.0
    out['present'] = bool(out['peak'] >= U_THRESH)
    return out


def eval_candidate(blocks):
    """Run the 3 patterns, compute the loss. Returns (loss, record)."""
    phi = build_phi(blocks)
    per_pattern = {}
    losses = []
    for tag in PATTERNS:
        fa, fb = FIRE[tag]
        series = run_pattern(phi, fa, fb)
        mets = {k: probe_metrics(s) for k, s in series.items()}
        q = ASSIGN[tag]
        presence = 1.0 if mets[q]['present'] else 0.0
        wrong = sum(1.0 for k, m in mets.items()
                    if k != q and m['present'])
        arr = mets[q]['first']
        if arr is None:
            timing = 1.0
        elif arr > WINDOW[1]:
            timing = min(1.0, (arr - WINDOW[1]) / 20.0)
        else:
            timing = 0.0
        lp = 2.0 * (1.0 - presence) + 0.5 * wrong + timing
        losses.append(lp)
        per_pattern[tag] = {'assigned': q, 'present': mets[q]['present'],
                            'arrival': arr, 'wrong': int(wrong),
                            'loss': lp}
    data_loss = float(np.mean(losses))
    V = np.asarray(blocks).reshape(NB, NB)
    gx, gy = np.gradient(V)
    tv = float(0.5 * (np.abs(gx).mean() + np.abs(gy).mean()))
    loss = data_loss + LAMBDA_TV * tv
    rec = {'loss': float(loss), 'data_loss': data_loss, 'tv': tv,
           'patterns': per_pattern}
    return loss, rec


# ---------------------------------------------------------------------------
# Fork-pool vectorized objective
# ---------------------------------------------------------------------------
_POOL = None


def _pool():
    global _POOL
    if _POOL is None:
        ctx = multiprocessing.get_context('fork')
        _POOL = ctx.Pool(POOL_WORKERS, maxtasksperchild=64)
    return _POOL


def _safe_eval(x):
    try:
        return eval_candidate(x)
    except Exception as exc:
        return FAIL_LOSS, {'loss': FAIL_LOSS, 'failed': True,
                           'error': f'{type(exc).__name__}: {exc}'}


def _eval_indexed(arg):
    """Pool worker: evaluate one candidate, keep its column index."""
    j, x = arg
    return j, _safe_eval(x)


# Parent-process progress tracker (workers are forked, so all logging
# happens here as results stream back).
_PROGRESS = {'n': 0, 't0': None, 'best': np.inf, 'gen': -1}


def _write_status(last_loss):
    p = _PROGRESS
    elapsed = time.time() - p['t0']
    eta = (elapsed / p['n']) * (EVALS_TOTAL - p['n']) if p['n'] else None
    status = {'updated': time.strftime('%Y-%m-%d %H:%M:%S'),
              'evals_done': p['n'], 'evals_total': EVALS_TOTAL,
              'generation': p['gen'],
              'gen_evals_done': p['n'] - p['gen'] * POP,
              'gen_evals_total': POP,
              'best_loss': None if np.isinf(p['best']) else p['best'],
              'last_loss': last_loss,
              'elapsed_min': round(elapsed / 60, 1),
              'eta_min': None if eta is None else round(eta / 60, 1)}
    tmp = os.path.join(FIG, 'rd_demux_status.json.tmp')
    with open(tmp, 'w') as fh:
        json.dump(status, fh, indent=2)
    os.replace(tmp, os.path.join(FIG, 'rd_demux_status.json'))


def objective(X):
    _PROGRESS['gen'] += 1
    if _PROGRESS['t0'] is None:
        _PROGRESS['t0'] = time.time()
    S = X.shape[1]
    xs = [(j, np.ascontiguousarray(X[:, j])) for j in range(S)]
    losses = np.empty(S)
    log_path = os.path.join(FIG, 'rd_demux_eval_log.jsonl')
    with open(log_path, 'a') as fh:
        for j, (loss, rec) in _pool().imap_unordered(_eval_indexed, xs,
                                                     chunksize=1):
            rec['t'] = time.time()
            rec['gen'] = _PROGRESS['gen']
            fh.write(json.dumps(rec) + '\n')
            fh.flush()
            losses[j] = loss
            _PROGRESS['n'] += 1
            new_best = loss < _PROGRESS['best']
            if new_best:
                _PROGRESS['best'] = loss
            _write_status(loss)
            if new_best or _PROGRESS['n'] % LOG_EVERY == 0:
                eta = _PROGRESS['n'] and (
                    (time.time() - _PROGRESS['t0']) / _PROGRESS['n']
                    * (EVALS_TOTAL - _PROGRESS['n']) / 60)
                print(f"[de] eval {_PROGRESS['n']}/{EVALS_TOTAL} "
                      f"(gen {_PROGRESS['gen']}, "
                      f"{_PROGRESS['n'] - _PROGRESS['gen'] * POP}/{POP}) "
                      f"last={loss:.4f} best={_PROGRESS['best']:.4f} "
                      f"eta={eta:.0f} min"
                      + ('  *new best*' if new_best else ''), flush=True)
    return losses


# ---------------------------------------------------------------------------
# Campaign
# ---------------------------------------------------------------------------
def protocol():
    print('[protocol] baseline (uniform phi0) ...', flush=True)
    t0 = time.time()
    loss, rec = eval_candidate(np.full(NB * NB, PHI0))
    proto = {'window': WINDOW, 't_flash': T_FLASH, 'tu_after': TU_AFTER,
             'assign': ASSIGN, 'design_region': [DR_X, DR_Y], 'nb': NB,
             'phi_bounds': [PHI_MIN, PHI_MAX], 'halo': HALO,
             'lambda_tv': LAMBDA_TV, 'baseline_loss': loss,
             'baseline_record': rec, 'wall_s': time.time() - t0}
    with open(os.path.join(FIG, 'rd_demux_protocol.json'), 'w') as fh:
        json.dump(proto, fh, indent=2)
    print(f'[protocol] baseline loss = {loss:.4f}', flush=True)
    for tag, p in rec['patterns'].items():
        print(f"  ({tag}) -> {p['assigned']}: present={p['present']}, "
              f"arr={p['arrival']}, wrong={p['wrong']}, loss={p['loss']:.3f}",
            flush=True)
    return proto


class Checkpoint:
    def __init__(self):
        self.gen = 0
        self.t0 = time.time()

    def __call__(self, xk, convergence):
        self.gen += 1
        state = {'generation': self.gen, 'best_x': [float(v) for v in xk],
                 'convergence': float(convergence),
                 'elapsed_s': time.time() - self.t0}
        with open(os.path.join(FIG, 'rd_demux_checkpoint.json'), 'w') as fh:
            json.dump(state, fh)
        print(f'[de] gen {self.gen} done, conv={convergence:.4f}, '
              f'elapsed={state["elapsed_s"] / 60:.0f} min', flush=True)


def train():
    print('[train] DE campaign start', flush=True)
    cb = Checkpoint()
    t0 = time.time()
    result = differential_evolution(
        objective, bounds=[(PHI_MIN, PHI_MAX)] * (NB * NB),
        strategy='best1bin', popsize=4, maxiter=MAXITER, tol=1e-4,
        mutation=(0.5, 1.0), recombination=0.7, seed=SEED,
        vectorized=True, workers=1, updating='deferred', polish=False,
        x0=np.full(NB * NB, PHI0), callback=cb, disp=False)
    wall = time.time() - t0
    print(f'[train] done: best loss {result.fun:.4f}, nfev={result.nfev}, '
          f'wall={wall / 3600:.1f} h', flush=True)

    # final evaluation + robustness of the best design
    best_loss, best_rec = _safe_eval(result.x)
    jit = []
    for dark_j in (DARK - 0.001, DARK + 0.001):
        globals()['DARK'] = dark_j
        try:
            _, r = _safe_eval(result.x)
            jit.append({'dark': dark_j, 'data_loss': r['data_loss']})
        finally:
            globals()['DARK'] = DARK
    out = {'best_loss': float(result.fun), 'best_x': [float(v) for v in result.x],
           'best_record': best_rec, 'nfev': int(result.nfev),
           'wall_s': wall, 'seed': SEED, 'jitter_dark': jit,
           'message': str(result.message)}
    with open(os.path.join(FIG, 'rd_demux_results.json'), 'w') as fh:
        json.dump(out, fh, indent=2)
    return out


def figures(proto, out):
    # loss history from eval log
    losses = []
    try:
        with open(os.path.join(FIG, 'rd_demux_eval_log.jsonl')) as fh:
            for line in fh:
                losses.append(json.loads(line)['loss'])
    except OSError:
        pass
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    if losses:
        axes[0].plot(losses, '.', ms=2)
        axes[0].axhline(proto['baseline_loss'], color='r', ls='--',
                        label=f"baseline {proto['baseline_loss']:.3f}")
        axes[0].set_xlabel('candidate eval')
        axes[0].set_ylabel('loss')
        axes[0].set_title('training history')
        axes[0].legend()
    phi = build_phi(out['best_x'])
    im = axes[1].imshow(phi.T, origin='lower', cmap='viridis',
                        vmin=PHI_MIN, vmax=PHI_MAX)
    for name, pos in PROBES.items():
        axes[1].plot(pos[0], pos[1], 'r^', ms=6)
    for name, pos in [('A', A_POS), ('B', B_POS)]:
        axes[1].plot(pos[0], pos[1], 'ws', ms=6)
    axes[1].set_title('best phi field')
    plt.colorbar(im, ax=axes[1])
    # per-pattern arrival comparison
    tags = PATTERNS
    base = [proto['baseline_record']['patterns'][t]['loss'] for t in tags]
    trained = [out['best_record']['patterns'][t]['loss'] for t in tags]
    x = np.arange(len(tags))
    axes[2].bar(x - 0.2, base, 0.4, label='baseline')
    axes[2].bar(x + 0.2, trained, 0.4, label='trained')
    axes[2].set_xticks(x, tags)
    axes[2].set_ylabel('pattern loss')
    axes[2].set_title('per-pattern loss')
    axes[2].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'rd_demux_history.png'), dpi=150)
    print('saved rd_demux_history.png (loss curve, phi map, per-pattern bars)')


if __name__ == '__main__':
    proto = protocol()
    out = train()
    figures(proto, out)
    print('[done]', flush=True)
