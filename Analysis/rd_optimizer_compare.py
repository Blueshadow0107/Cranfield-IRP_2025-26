"""
Compare black-box optimisers on a T-junction routing task.

Task: fixed walls, single input A at the base of the stem.
Optimise phi(x,y) so the pulse reaches the LEFT output probe and NOT the
RIGHT one.  The symmetric task (route to RIGHT) can be run by flipping the
target.

Optimisers:
  - random search (baseline)
  - scipy differential_evolution
  - scipy dual_annealing
  - CMA-ES (pycma)

Dimension: NB*NB = 16 for the default 4x4 design grid.
Budget: small (30-50 evals) so the comparison finishes quickly.
"""
import json
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, dual_annealing
import cma

# Import the T-junction machinery (uses max-probe readout)
from rd_tjunc_router import (
    NB, PHI_MIN, PHI_MAX, build_phi, run_pattern, probe_metrics
)

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG, exist_ok=True)

BUDGET = 20
SEED = 7
TARGET = 'left'   # 'left' or 'right'


def objective(blocks, target=TARGET):
    """
    Scalar objective for routing input A to the chosen target probe.
    Continuous (soft) loss based on peak amplitude; easier for optimisers
    than a hard threshold.
    """
    phi = build_phi(blocks)
    t, left, right = run_pattern(phi, True, False)
    if target == 'left':
        assigned_peak = left.max()
        wrong_peak = right.max()
    else:
        assigned_peak = right.max()
        wrong_peak = left.max()
    # want assigned high, wrong low
    loss = max(0.0, 0.7 - assigned_peak) / 0.7 + wrong_peak / 0.7
    # tiny TV penalty
    V = np.asarray(blocks).reshape(NB, NB)
    gx, gy = np.gradient(V)
    tv = 0.5 * (np.abs(gx).mean() + np.abs(gy).mean())
    return loss + 0.01 * tv


def hard_loss(blocks, target=TARGET):
    """Hard thresholded loss (for final reporting)."""
    phi = build_phi(blocks)
    t, left, right = run_pattern(phi, True, False)
    ml, mr = probe_metrics(left), probe_metrics(right)
    if target == 'left':
        presence = 1.0 if ml['present'] else 0.0
        wrong = 1.0 if mr['present'] else 0.0
    else:
        presence = 1.0 if mr['present'] else 0.0
        wrong = 1.0 if ml['present'] else 0.0
    return 2.0 * (1.0 - presence) + 1.0 * wrong


def random_search(obj, budget=BUDGET):
    rng = np.random.default_rng(SEED)
    history = []
    best_x, best_f = None, 1e9
    for i in range(budget):
        x = rng.uniform(PHI_MIN, PHI_MAX, NB * NB)
        f = obj(x)
        history.append({'f': float(f), 't': time.time()})
        if f < best_f:
            best_f, best_x = f, x.copy()
        print(f"  rand eval {i+1:2d}/{budget}: f={f:.3f} best={best_f:.3f}", flush=True)
    return best_x, best_f, history


def run_de(obj, budget=BUDGET):
    history = []
    def cb(xk, convergence):
        f = float(obj(xk))
        history.append({'f': f, 't': time.time()})
        print(f"  DE eval {len(history):2d}/{budget}: f={f:.3f}", flush=True)
    bounds = [(PHI_MIN, PHI_MAX)] * (NB * NB)
    # popsize=1 -> initial population = 16; maxiter=1 -> one generation = 16 more
    result = differential_evolution(
        obj, bounds, maxiter=1,
        popsize=1,
        mutation=(0.5, 1.0), recombination=0.7,
        seed=SEED, workers=1, polish=False,
        updating='immediate',
        callback=cb,
        tol=0.0, atol=0.0
    )
    history = history[:budget]
    # best x from truncated history
    best_idx = int(np.argmin([h['f'] for h in history]))
    best_x = result.x if result.fun <= history[best_idx]['f'] else None
    return result.x, result.fun, history


def run_da(obj, budget=BUDGET):
    history = []
    def cb(x, f, context):
        history.append({'f': float(f), 't': time.time()})
        print(f"  DA eval {len(history):2d}/{budget}: f={f:.3f}", flush=True)
    bounds = [(PHI_MIN, PHI_MAX)] * (NB * NB)
    result = dual_annealing(obj, bounds, maxfun=budget, seed=SEED, callback=cb)
    return result.x, result.fun, history


def run_cma(obj, budget=BUDGET):
    x0 = np.full(NB * NB, (PHI_MIN + PHI_MAX) / 2.0)
    sigma0 = (PHI_MAX - PHI_MIN) / 4.0
    opts = {
        'bounds': [[PHI_MIN] * (NB * NB), [PHI_MAX] * (NB * NB)],
        'maxfevals': budget,
        'seed': SEED,
        'verbose': -9,
        'verb_log': 0,
    }
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    history = []
    while not es.stop():
        solutions = es.ask()
        fitnesses = [obj(x) for x in solutions]
        for f in fitnesses:
            history.append({'f': float(f), 't': time.time()})
            print(f"  CMA eval {len(history):2d}/{budget}: f={f:.3f}", flush=True)
        es.tell(solutions, fitnesses)
    xbest = es.result.xfavorite
    fbest = es.result.fbest
    return xbest, fbest, history[:budget]


def compare():
    obj = objective
    print(f"\n=== optimizer comparison: route input A to {TARGET} (budget={BUDGET}) ===")
    results = {}
    for name, fn in [('random', random_search), ('DE', run_de),
                     ('dual_annealing', run_da), ('CMA-ES', run_cma)]:
        t0 = time.time()
        x, f, hist = fn(obj, BUDGET)
        elapsed = time.time() - t0
        hloss = hard_loss(x)
        results[name] = {'best_f': float(f), 'best_x': x.tolist(),
                         'hard_loss': float(hloss),
                         'history': hist, 'time': elapsed}
        print(f"{name:16s} soft={f:.3f}  hard={hloss:.3f}  time={elapsed:.1f}s  n_eval={len(hist)}")

    with open(os.path.join(FIG, f'rd_optimizer_compare_{TARGET}.json'), 'w') as fh:
        json.dump(results, fh, indent=2)

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, d in results.items():
        fs = [h['f'] for h in d['history']]
        best_so_far = np.minimum.accumulate(fs)
        ax.plot(np.arange(1, len(best_so_far) + 1), best_so_far, '.-', label=name)
    ax.set_xlabel('function evaluations')
    ax.set_ylabel('best soft loss so far')
    ax.set_title(f'optimizer comparison: route to {TARGET} (NB={NB}, budget={BUDGET})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, f'rd_optimizer_compare_{TARGET}.png'), dpi=150)
    print(f"[saved] {FIG}/rd_optimizer_compare_{TARGET}.png")
    return results


if __name__ == '__main__':
    compare()
