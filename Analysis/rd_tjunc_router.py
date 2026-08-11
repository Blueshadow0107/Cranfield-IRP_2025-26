"""
T-junction routing prototype with fixed walls and trainable phi(x,y).

Geometry
--------
200x200 domain. Vertical stem W=16 from y=40 up to y=100. Horizontal
bar W=16 from x=60 to x=140 at y=100. Output probes at left (40,100)
and right (160,100). Input ports A and B at the base of the stem:
A centred at (100,30), B not used in single-input tests; for pattern
tests A=(92,30), B=(108,30).

Control space
-------------
Design region x=[60,140], y=[60,140] (around the junction), 4x4 or 8x8
bilinear phi blocks, clipped to [PHI_MIN, PHI_MAX]. Background phi0
holds the medium excitable.

Task
----
Pattern "10" (A only) -> left probe fires, right does not.
Pattern "01" (B only) -> right probe fires, left does not.
Pattern "11" (A+B)    -> neither (or both, set by target).

This script supports hand-designed phi tests and small optimizer runs.
"""
import json
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import differential_evolution
from rd_core import RDSubstrate

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG, exist_ok=True)

# Physics (locked)
F, EPS, Q = 1.4375, 0.05014844822490394, 0.002
PHI0 = 0.010
DARK = 0.002
U_STAR = 0.0030821
DT = 0.05
T_FLASH = 4.0
TU_AFTER = 20.0
U_THRESH = 0.5

# Domain (prototype: smaller for speed)
NX, NY = 120, 120

# Geometry (walls)
STEM_CX, STEM_W = 60, 14
STEM_Y0, STEM_Y1 = 10, 60
BAR_Y, BAR_W = 60, 14
BAR_X0, BAR_X1 = 35, 85

# Input dark spots (at base of stem)
SPOT_R = 10
A_POS = (54, 18)
B_POS = (66, 18)

# Probes (inside the horizontal bar)
PROBE_R = 6
L_POS = (42, 60)
R_POS = (78, 60)

# Design region
DR_X, DR_Y = (35, 85), (35, 85)
NB = 4
PHI_MIN, PHI_MAX = 0.010, 0.040


def build_walls():
    wall = np.ones((NX, NY), bool)
    # vertical stem
    x0 = STEM_CX - STEM_W // 2
    x1 = STEM_CX + STEM_W // 2
    wall[x0:x1, STEM_Y0:STEM_Y1] = False
    # horizontal bar
    y0 = BAR_Y - BAR_W // 2
    y1 = BAR_Y + BAR_W // 2
    wall[BAR_X0:BAR_X1, y0:y1] = False
    return wall


WALL = build_walls()


def spot_mask(pos, r=SPOT_R):
    X, Y = np.mgrid[0:NX, 0:NY].astype(float)
    return ((X - pos[0])**2 + (Y - pos[1])**2) < r**2


def probe_mask(pos, r=PROBE_R):
    X, Y = np.mgrid[0:NX, 0:NY].astype(float)
    return (((X - pos[0])**2 + (Y - pos[1])**2) < r**2) & ~WALL


A_MASK = spot_mask(A_POS)
B_MASK = spot_mask(B_POS)
L_MASK = probe_mask(L_POS)
R_MASK = probe_mask(R_POS)


def build_phi(blocks):
    """blocks: flat array of NB*NB values -> bilinear interpolated phi field."""
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
    field[WALL] = PHI0  # walls not part of design
    return np.clip(field, PHI_MIN, PHI_MAX)


def run_pattern(phi, fire_a, fire_b):
    rd = RDSubstrate(nx=NX, ny=NY, dt=DT, kinetics='oregonator',
                     f=F, eps=EPS, clamp_rest=(U_STAR, U_STAR))
    rd.set_walls(WALL)
    rd.set_phi(phi)
    rd.u[:] = U_STAR
    rd.v[:] = U_STAR
    phi_dark = phi.copy()
    if fire_a:
        phi_dark[A_MASK] = DARK
    if fire_b:
        phi_dark[B_MASK] = DARK
    rd.set_phi(phi_dark)
    rd.run(int(T_FLASH / DT))
    rd.set_phi(phi)
    nsteps = int(TU_AFTER / DT)
    t = np.arange(nsteps) * DT
    left = np.empty(nsteps)
    right = np.empty(nsteps)
    for n in range(nsteps):
        rd.run(1)
        left[n] = rd.u[L_MASK].max() if L_MASK.any() else 0.0
        right[n] = rd.u[R_MASK].max() if R_MASK.any() else 0.0
    return t, left, right


def probe_metrics(series):
    above = series >= U_THRESH
    first = float(np.argmax(above) * DT) if above.any() else None
    peak = float(series.max())
    present = peak >= U_THRESH
    return {'first': first, 'peak': peak, 'present': present}


def eval_candidate(blocks, verbose=False):
    phi = build_phi(blocks)
    # targets: 10 -> left, 01 -> right, 11 -> neither
    targets = {'10': 'L', '01': 'R', '11': 'N'}
    fire = {'10': (True, False), '01': (False, True), '11': (True, True)}
    losses = []
    detail = {}
    for tag, tgt in targets.items():
        fa, fb = fire[tag]
        t, left, right = run_pattern(phi, fa, fb)
        ml, mr = probe_metrics(left), probe_metrics(right)
        detail[tag] = {'left': ml, 'right': mr}
        if tgt == 'L':
            presence = 1.0 if ml['present'] else 0.0
            wrong = (1.0 if mr['present'] else 0.0)
            timing = 0.0
        elif tgt == 'R':
            presence = 1.0 if mr['present'] else 0.0
            wrong = (1.0 if ml['present'] else 0.0)
            timing = 0.0
        else:  # neither
            presence = 0.0
            wrong = (1.0 if ml['present'] else 0.0) + (1.0 if mr['present'] else 0.0)
            timing = 0.0
        lp = 2.0 * (1.0 - presence) + 1.0 * wrong + timing
        losses.append(lp)
        if verbose:
            print(f"  {tag}->{tgt} L={ml['present']}({ml['peak']:.3f}) "
                  f"R={mr['present']}({mr['peak']:.3f}) loss={lp:.2f}")
    data_loss = float(np.mean(losses))
    V = np.asarray(blocks).reshape(NB, NB)
    gx, gy = np.gradient(V)
    tv = float(0.5 * (np.abs(gx).mean() + np.abs(gy).mean()))
    loss = data_loss + 0.02 * tv
    return loss, {'loss': loss, 'data_loss': data_loss, 'tv': tv,
                  'detail': detail, 'blocks': blocks.tolist()}


def hand_tests():
    print("=== hand-designed phi tests ===")
    tests = {
        'uniform_phi0': np.full(NB * NB, PHI0),
        'left_block_high': np.full(NB * NB, PHI0),
        'right_block_high': np.full(NB * NB, PHI0),
        'top_block_high': np.full(NB * NB, PHI0),
    }
    # left half high -> should block left branch, route right?
    arr = np.full((NB, NB), PHI0)
    arr[:NB // 2, :] = PHI_MAX
    tests['left_block_high'] = arr.flatten()
    # right half high
    arr = np.full((NB, NB), PHI0)
    arr[NB // 2:, :] = PHI_MAX
    tests['right_block_high'] = arr.flatten()
    # top half high (near junction)
    arr = np.full((NB, NB), PHI0)
    arr[:, NB // 2:] = PHI_MAX
    tests['top_block_high'] = arr.flatten()

    results = {}
    for name, blocks in tests.items():
        print(f"\n{name}")
        loss, rec = eval_candidate(blocks, verbose=True)
        print(f"  total loss={loss:.3f}")
        results[name] = rec
    with open(os.path.join(FIG, 'rd_tjunc_hand_tests.json'), 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f"[saved] {FIG}/rd_tjunc_hand_tests.json")


def random_baseline(n=20):
    rng = np.random.default_rng(0)
    best = (1e9, None)
    history = []
    print(f"\n=== random baseline ({n} evals) ===")
    for i in range(n):
        blocks = rng.uniform(PHI_MIN, PHI_MAX, NB * NB)
        t0 = time.time()
        loss, rec = eval_candidate(blocks)
        dt = time.time() - t0
        history.append({'loss': loss, 'data_loss': rec['data_loss'], 'time': dt})
        if loss < best[0]:
            best = (loss, blocks.copy())
        print(f"  {i:3d} loss={loss:.3f}  best={best[0]:.3f}  t={dt:.1f}s")
    print(f"random best loss={best[0]:.3f}")
    with open(os.path.join(FIG, 'rd_tjunc_random_baseline.json'), 'w') as fh:
        json.dump({'best_loss': best[0], 'history': history}, fh, indent=2)
    return best


def de_run(maxiter=3, pop=4 * NB * NB):
    print(f"\n=== DE run (pop={pop}, maxiter={maxiter}) ===")
    bounds = [(PHI_MIN, PHI_MAX)] * (NB * NB)
    result = differential_evolution(
        lambda x: eval_candidate(x)[0],
        bounds,
        maxiter=maxiter,
        popsize=pop // (NB * NB) or 1,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42,
        workers=-1,  # uses multiprocessing.Pool internally
        polish=False,
        updating='deferred'
    )
    print(f"DE best loss={result.fun:.3f}")
    loss, rec = eval_candidate(result.x, verbose=True)
    with open(os.path.join(FIG, 'rd_tjunc_de_result.json'), 'w') as fh:
        json.dump({'x': result.x.tolist(), 'fun': result.fun,
                   'nfev': result.nfev, 'detail': rec}, fh, indent=2)
    plot_phi(result.x, 'rd_tjunc_de_phi.png')
    return result.x, result.fun


def plot_phi(blocks, filename):
    phi = build_phi(blocks)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(phi.T, origin='lower', vmin=PHI_MIN, vmax=PHI_MAX,
                   cmap='viridis', extent=[0, NX, 0, NY])
    # overlay walls
    wall_x, wall_y = np.where(WALL)
    ax.scatter(wall_x, wall_y, c='k', s=1, alpha=0.3)
    ax.scatter([A_POS[0], B_POS[0]], [A_POS[1], B_POS[1]], c='r', s=50, marker='o')
    ax.scatter([L_POS[0], R_POS[0]], [L_POS[1], R_POS[1]], c='g', s=50, marker='s')
    plt.colorbar(im, ax=ax, label='phi')
    ax.set_title('trainable phi field')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, filename), dpi=150)
    print(f"[saved] {FIG}/{filename}")


if __name__ == '__main__':
    hand_tests()
    random_baseline(n=5)
    # de_run(maxiter=2, pop=32)
