"""
Trainable wall geometry for two-input T-junction routing.

Geometry representation:
- 120x120 domain.
- Vertical stem, width fixed, horizontal position trainable.
- Horizontal bar at fixed y, left and right branches.
- A small square obstacle near the junction, position and size trainable.
- Inputs: A and B are dark spots at the left and right sides of the stem base.

Target: input A (left side of stem) routes to the right output probe,
        input B (right side of stem) routes to the left output probe.

The idea is that an asymmetric obstacle at the junction can steer a pulse
based on which side of the stem it enters, something a static phi field
cannot do when both inputs are in the same stem.
"""
import json, os, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cma
from rd_core import RDSubstrate

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')

# Physics (locked)
F, EPS, Q = 1.4375, 0.05014844822490394, 0.002
PHI0 = 0.010
DARK = 0.002
U_STAR = 0.0030821
DT = 0.05
T_FLASH = 4.0
TU_AFTER = 25.0
U_THRESH = 0.5

NX, NY = 120, 120

# Fixed probe positions
L_POS = (35, 60)
R_POS = (85, 60)
PROBE_R = 6

# Input dark spots: A on left side of stem base, B on right side
SPOT_R = 8


def spot_mask(pos, r=SPOT_R):
    X, Y = np.mgrid[0:NX, 0:NY].astype(float)
    return ((X - pos[0])**2 + (Y - pos[1])**2) < r**2


def probe_mask(pos, r=PROBE_R):
    X, Y = np.mgrid[0:NX, 0:NY].astype(float)
    return (((X - pos[0])**2 + (Y - pos[1])**2) < r**2)


L_MASK = probe_mask(L_POS)
R_MASK = probe_mask(R_POS)


def build_walls(params):
    """
    params: [stem_cx, stem_width, stem_y1, junction_y, bar_width,
             obs_cx, obs_cy, obs_half]
    Returns boolean wall mask (True = wall).
    """
    stem_cx, stem_width, stem_y1, junction_y, bar_width, obs_cx, obs_cy, obs_half = params
    stem_cx = int(round(stem_cx))
    stem_width = int(round(stem_width))
    stem_y1 = int(round(stem_y1))
    junction_y = int(round(junction_y))
    bar_width = int(round(bar_width))
    obs_cx = int(round(obs_cx))
    obs_cy = int(round(obs_cy))
    obs_half = int(round(obs_half))

    wall = np.ones((NX, NY), bool)
    # vertical stem from y=stem_y1 up to junction_y
    x0 = stem_cx - stem_width // 2
    x1 = stem_cx + stem_width // 2
    wall[x0:x1, stem_y1:junction_y] = False
    # horizontal bar across junction_y
    y0 = junction_y - bar_width // 2
    y1 = junction_y + bar_width // 2
    wall[10:110, y0:y1] = False
    # central obstacle
    if obs_half > 0:
        ox0 = max(0, obs_cx - obs_half)
        ox1 = min(NX, obs_cx + obs_half)
        oy0 = max(0, obs_cy - obs_half)
        oy1 = min(NY, obs_cy + obs_half)
        wall[ox0:ox1, oy0:oy1] = True
    return wall


def input_positions(params):
    stem_cx, stem_width, stem_y1, *_ = params
    # A left side, B right side of stem base
    left_x = stem_cx - stem_width // 4
    right_x = stem_cx + stem_width // 4
    y = stem_y1 + 6
    return (left_x, y), (right_x, y)


def run_pattern(phi, wall, fire_a, fire_b, a_pos, b_pos):
    rd = RDSubstrate(nx=NX, ny=NY, dt=DT, kinetics='oregonator',
                     f=F, eps=EPS, clamp_rest=(U_STAR, U_STAR))
    rd.set_walls(wall)
    rd.set_phi(phi)
    rd.u[:] = U_STAR
    rd.v[:] = U_STAR
    phi_dark = phi.copy()
    if fire_a:
        phi_dark[spot_mask(a_pos)] = DARK
    if fire_b:
        phi_dark[spot_mask(b_pos)] = DARK
    rd.set_phi(phi_dark)
    rd.run(int(T_FLASH / DT))
    rd.set_phi(phi)
    nsteps = int(TU_AFTER / DT)
    left = np.empty(nsteps)
    right = np.empty(nsteps)
    for n in range(nsteps):
        rd.run(1)
        left[n] = rd.u[L_MASK].max() if L_MASK.any() else 0.0
        right[n] = rd.u[R_MASK].max() if R_MASK.any() else 0.0
    return left, right


def eval_geometry(params, verbose=False):
    wall = build_walls(params)
    phi = np.full((NX, NY), PHI0)
    a_pos, b_pos = input_positions(params)

    # A -> right target
    l_a, r_a = run_pattern(phi, wall, True, False, a_pos, b_pos)
    # B -> left target
    l_b, r_b = run_pattern(phi, wall, False, True, a_pos, b_pos)

    a_to_right = r_a.max() >= U_THRESH and l_a.max() < U_THRESH
    b_to_left = l_b.max() >= U_THRESH and r_b.max() < U_THRESH

    hard_loss = 0.0
    hard_loss += 0.0 if a_to_right else 1.0
    hard_loss += 0.0 if b_to_left else 1.0

    # soft loss: encourage correct peaks and suppress wrong peaks
    soft = 0.0
    soft += (1.0 - min(r_a.max() / 0.7, 1.0)) + max(l_a.max() - 0.05, 0.0) / 0.7
    soft += (1.0 - min(l_b.max() / 0.7, 1.0)) + max(r_b.max() - 0.05, 0.0) / 0.7
    soft /= 4.0

    loss = hard_loss + 0.5 * soft
    if verbose:
        print(f"  A: L={l_a.max():.3f} R={r_a.max():.3f}  B: L={l_b.max():.3f} R={r_b.max():.3f}  loss={loss:.3f}")
    return loss, {
        'loss': loss, 'hard_loss': hard_loss, 'soft_loss': soft,
        'A': {'L': float(l_a.max()), 'R': float(r_a.max())},
        'B': {'L': float(l_b.max()), 'R': float(r_b.max())},
        'params': params.tolist()
    }


def random_baseline(n=20):
    print(f'=== random baseline ({n} evals) ===')
    # bounds: stem_cx, stem_width, stem_y1, junction_y, bar_width, obs_cx, obs_cy, obs_half
    lows = np.array([45, 10, 10, 45, 8, 40, 45, 0])
    highs = np.array([75, 22, 25, 65, 18, 80, 75, 8])
    rng = np.random.default_rng(7)
    best = (1e9, None)
    for i in range(n):
        p = rng.uniform(lows, highs)
        loss, rec = eval_geometry(p)
        if loss < best[0]:
            best = (loss, p.copy())
        print(f"  {i:2d} loss={loss:.3f} hard={rec['hard_loss']:.0f}")
    print(f'random best loss={best[0]:.3f}')
    return best


def cmaes_run(maxiter=8):
    print('\n=== CMA-ES ===')
    x0 = np.array([60, 14, 15, 60, 12, 60, 60, 3])
    sigma0 = 3.0
    lows = np.array([45, 10, 10, 45, 8, 40, 45, 0])
    highs = np.array([75, 22, 25, 65, 18, 80, 75, 8])
    opts = cma.CMAOptions()
    opts.set('bounds', [lows.tolist(), highs.tolist()])
    opts.set('maxiter', maxiter)
    opts.set('verbose', -1)
    opts.set('verb_log', 0)
    res = cma.fmin(lambda x: eval_geometry(x)[0], x0, sigma0, opts)
    best = res[0]
    loss, rec = eval_geometry(best, verbose=True)
    print(f'CMA-ES best loss={loss:.3f} hard={rec["hard_loss"]:.0f}')
    return best, rec


def plot_geometry(params, filename):
    wall = build_walls(params)
    a_pos, b_pos = input_positions(params)
    fig, ax = plt.subplots(figsize=(5, 5))
    wall_x, wall_y = np.where(wall)
    ax.scatter(wall_x, wall_y, c='k', s=1, alpha=0.5)
    ax.scatter([a_pos[0], b_pos[0]], [a_pos[1], b_pos[1]], c='r', s=80, marker='o')
    ax.scatter([L_POS[0], R_POS[0]], [L_POS[1], R_POS[1]], c='g', s=80, marker='s')
    ax.set_xlim(0, NX)
    ax.set_ylim(0, NY)
    ax.set_aspect('equal')
    ax.set_title('trainable T-junction geometry')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, filename), dpi=150)
    print(f'[saved] {FIG}/{filename}')


def main():
    random_baseline(n=15)
    best_params, rec = cmaes_run(maxiter=8)
    with open(os.path.join(FIG, 'rd_tjunc_geometry_train.json'), 'w') as fh:
        json.dump(rec, fh, indent=2)
    plot_geometry(best_params, 'rd_tjunc_geometry_train.png')


if __name__ == '__main__':
    main()
