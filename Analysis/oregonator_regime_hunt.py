"""
Oregonator excitable-regime hunt (point kinetics) + candidate validation.

Motivation
----------
At the baseline (f=1.4, eps=0.05, q=0.002, phi=0) the Tyson-Fife
Oregonator rest state is an UNSTABLE node -- the medium is a relaxation
oscillator, useless as a substrate that must wait quietly for input
pulses.  A scan of eps alone found the model goes from oscillatory
straight to subexcitable, but phi (light suppression) and f were never
scanned jointly with eps.  Photosensitive BZ experiments hold the medium
excitable via background illumination, so an excitable island in
(f, eps, phi) space should exist.  This script looks for it.

Point kinetics (no diffusion):
    du/dt = (1/eps) * [u - u^2 - (f v + phi) (u - q)/(u + q)]
    dv/dt = u - v

Classification per grid point (f, eps, phi):
  OSCILLATORY      rest fixed point is linearly unstable (max Re eig > 0),
                   or a perturbed trajectory never returns to rest;
  STABLE-EXCITABLE stable rest state AND a finite perturbation
                   (u0=0.2, v=v*) fires a pulse (u_max > 0.5) followed by
                   return to rest; if u0=0.2 fails but u0=0.5 succeeds the
                   point is excitable with a high threshold (recorded);
  SUBEXCITABLE     stable rest state, no pulse for u0=0.2 or u0=0.5.

The pulse tests are integrated UNCLIPPED with scipy LSODA (stiff,
rtol=1e-7) -- no np.clip anywhere.

Usage:
    python oregonator_regime_hunt.py scan        # grid scan + figure
    python oregonator_regime_hunt.py propagate   # channel tests for the
                                                 # candidates in the JSON

Outputs:
    figures/oregonator_regime_scan.npz        (raw grid, for replotting)
    figures/oregonator_regime_map.png         (phase diagram, 4 phi slices)
    figures/oregonator_regime_candidates.json (candidates + speeds)
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.integrate import solve_ivp

Q = 0.002
F_GRID = np.linspace(0.5, 3.0, 25)
EPS_GRID = np.logspace(np.log10(0.005), np.log10(0.2), 25)
PHI_GRID = np.linspace(0.0, 0.08, 17)

OSC, EXC, EXC_HT, SUB = 2, 1, 3, 0     # class codes
CLASS_NAMES = {OSC: 'oscillatory', EXC: 'excitable',
               EXC_HT: 'excitable_high_threshold', SUB: 'subexcitable'}

SCAN_NPZ = 'figures/oregonator_regime_scan.npz'
MAP_PNG = 'figures/oregonator_regime_map.png'
CAND_JSON = 'figures/oregonator_regime_candidates.json'


# ---------------------------------------------------------------------------
# Fixed point and linear stability
# ---------------------------------------------------------------------------
def F_uonly(u, f, phi):
    """u-nullcline intersection condition with v = u (dv/dt = 0)."""
    return u - u**2 - (f * u + phi) * (u - Q) / (u + Q)


def rest_state(f, phi):
    """Smallest positive root of F_uonly (= physical rest state)."""
    ugrid = np.concatenate(([0.0], np.logspace(-7, np.log10(1.2), 4000)))
    vals = F_uonly(ugrid, f, phi)
    for i in range(len(ugrid) - 1):
        if vals[i] > 0.0 and vals[i + 1] <= 0.0:
            return brentq(F_uonly, ugrid[i], ugrid[i + 1], args=(f, phi),
                          xtol=1e-14, rtol=1e-12)
    return None


def rest_eigenvalues(u_star, f, eps, phi):
    """Jacobian eigenvalues at (u*, v*=u*)."""
    g = (u_star - Q) / (u_star + Q)
    dg = 2.0 * Q / (u_star + Q)**2
    s = f * u_star + phi
    Fu = (1.0 - 2.0 * u_star - s * dg) / eps
    Fv = -(f * g) / eps
    tr = Fu - 1.0
    det = -Fu - Fv
    disc = tr * tr - 4.0 * det
    if disc >= 0.0:
        r = np.sqrt(disc)
        return np.array([0.5 * (tr + r), 0.5 * (tr - r)])
    r = np.sqrt(-disc)
    return np.array([0.5 * tr + 0.5j * r, 0.5 * tr - 0.5j * r])


# ---------------------------------------------------------------------------
# Pulse test (unclipped, stiff solver)
# ---------------------------------------------------------------------------
def rhs(t, y, f, eps, phi):
    u, v = y
    return [(u - u * u - (f * v + phi) * (u - Q) / (u + Q)) / eps, u - v]


def pulse_test(f, eps, phi, u_star, u0):
    """Integrate from (u0, v*=u*) for 500 t.u.; return (fired, returned)."""
    sol = solve_ivp(rhs, (0.0, 500.0), [u0, u_star], args=(f, eps, phi),
                    method='LSODA', rtol=1e-7, atol=1e-10,
                    t_eval=np.linspace(0.0, 500.0, 2501))
    u = sol.y[0]
    fired = bool(u.max() > 0.5)
    tail = u[int(0.8 * len(u)):]           # last 100 t.u.
    returned = bool(u[-1] < 0.05 and tail.max() < 0.3)
    return fired, returned


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------
def scan():
    nf, ne, np_ = len(F_GRID), len(EPS_GRID), len(PHI_GRID)
    cls = np.full((nf, ne, np_), -1, dtype=int)
    ustars = np.full((nf, ne, np_), np.nan)
    maxre = np.full((nf, ne, np_), np.nan)
    n_stable = 0
    for i, f in enumerate(F_GRID):
        for j, eps in enumerate(EPS_GRID):
            for k, phi in enumerate(PHI_GRID):
                us = rest_state(f, phi)
                if us is None:
                    cls[i, j, k] = OSC
                    continue
                ustars[i, j, k] = us
                ev = rest_eigenvalues(us, f, eps, phi)
                maxre[i, j, k] = float(ev.real.max())
                if ev.real.max() > 1e-9:
                    cls[i, j, k] = OSC
                    continue
                n_stable += 1
                fired, returned = pulse_test(f, eps, phi, us, 0.2)
                if fired and returned:
                    cls[i, j, k] = EXC
                elif fired:                 # never came back -> oscillator
                    cls[i, j, k] = OSC
                else:
                    fired2, returned2 = pulse_test(f, eps, phi, us, 0.5)
                    if fired2 and returned2:
                        cls[i, j, k] = EXC_HT
                    elif fired2:
                        cls[i, j, k] = OSC
                    else:
                        cls[i, j, k] = SUB
        print(f'f={f:.3f} done '
              f'(exc={np.sum(cls[i] == EXC)}, ht={np.sum(cls[i] == EXC_HT)}, '
              f'osc={np.sum(cls[i] == OSC)}, sub={np.sum(cls[i] == SUB)})',
              flush=True)
    print(f'stable fixed points integrated: {n_stable}')
    np.savez(SCAN_NPZ, cls=cls, ustars=ustars, maxre=maxre,
             f_grid=F_GRID, eps_grid=EPS_GRID, phi_grid=PHI_GRID)
    print(f'Saved {SCAN_NPZ}')
    return cls


def plot_map():
    d = np.load(SCAN_NPZ)
    cls, f_g, e_g, p_g = d['cls'], d['f_grid'], d['eps_grid'], d['phi_grid']
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(['#9ecae1', '#31a354', '#fb6a4a', '#c7e9c0'])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), sharey=True)
    for ax, phi0 in zip(axes, [0.0, 0.02, 0.04, 0.06]):
        k = int(np.argmin(np.abs(p_g - phi0)))
        im = ax.pcolormesh(f_g, e_g, cls[:, :, k].T, cmap=cmap, norm=norm,
                           shading='nearest')
        ax.set_yscale('log')
        ax.set_xlabel('f')
        ax.set_title(f'phi = {p_g[k]:.3f}')
        ax.axvline(1.4, color='k', ls=':', lw=0.8)
    axes[0].set_ylabel('eps')
    cbar = fig.colorbar(im, ax=axes, ticks=[0, 1, 2, 3], shrink=0.85)
    cbar.ax.set_yticklabels(['subexcitable', 'excitable', 'oscillatory',
                             'excitable (high thr.)'])
    fig.suptitle('Oregonator point-kinetics regime map (q=0.002): '
                 'rest-state stability + pulse test', fontsize=13)
    fig.tight_layout()
    fig.savefig(MAP_PNG, dpi=150)
    print(f'Saved {MAP_PNG}')


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------
def pick_candidates():
    d = np.load(SCAN_NPZ)
    cls, us, f_g, e_g, p_g = (d['cls'], d['ustars'], d['f_grid'],
                              d['eps_grid'], d['phi_grid'])
    cands = []
    for i, f in enumerate(f_g):
        for j, eps in enumerate(e_g):
            for k, phi in enumerate(p_g):
                if cls[i, j, k] in (EXC, EXC_HT):
                    # distance to literature neighbourhood (f~1.4, eps~0.05)
                    dist = np.sqrt(((f - 1.4) / 0.5)**2
                                   + (np.log10(eps / 0.05) / 0.5)**2)
                    cands.append(dict(f=float(f), eps=float(eps),
                                      phi=float(phi), q=Q,
                                      u_star=float(us[i, j, k]),
                                      max_re_eig=float(d['maxre'][i, j, k]),
                                      cls=int(cls[i, j, k]),
                                      class_name=CLASS_NAMES[int(cls[i, j, k])],
                                      dist_lit=float(dist)))
    cands.sort(key=lambda c: c['dist_lit'])
    return cands


# ---------------------------------------------------------------------------
# Propagation validation with the de-hacked rd_core
# ---------------------------------------------------------------------------
def propagate(cand, nx=300, ny=40, width=12, dt=0.05, nsteps=6000):
    """Fire one pulse into a walled channel; measure travel and speed.

    The channel is initialised AT the rest state (u*, v*=u*) -- starting
    from (0,0) is wrong: du/dt(0) = phi/eps > 0, and the homogeneous
    transient toward u* can itself cross the excitation threshold at
    small eps (observed: global ignition at eps=0.0199).

    Returns dict with travelled flag (pulse crosses x = port_end + 100),
    speed in cells/t.u. between the two probes, number of pulses seen at
    the near probe (must be 1 -- a train means the wake re-fires), and a
    no-stimulus control (medium must stay at rest)."""
    from rd_core import RDSubstrate
    cy = ny // 2
    us = cand['u_star']

    def build():
        rd = RDSubstrate(nx=nx, ny=ny, dt=dt, kinetics='oregonator',
                         eps=cand['eps'], f=cand['f'])
        rd.set_phi(cand['phi'])
        wall = np.ones((nx, ny), bool)
        wall[:, cy - width // 2:cy + width // 2] = False
        rd.set_walls(wall)
        rd.u[~wall] = us
        rd.v[~wall] = us
        return rd, wall

    rd, wall = build()
    port = np.zeros((nx, ny), bool)
    port[0:18, cy - width // 2:cy + width // 2] = True
    rd.add_port('in', port)
    X, Y = np.mgrid[0:nx, 0:ny]

    def probe_mask(x0):
        return ((X - x0)**2 + (Y - cy)**2 <= 4) & ~wall

    rd.add_probe('p1', probe_mask(80))
    rd.add_probe('p2', probe_mask(180))
    rd.fire('in', value=0.8, v_value=0.2, duration=30)
    data = rd.run(nsteps)

    def crossings(s):
        up = (s[:-1] < 0.5) & (s[1:] >= 0.5)
        idx = np.nonzero(up)[0]
        return idx

    c1, c2 = crossings(data['p1']), crossings(data['p2'])
    t1 = float(c1[0] * dt) if len(c1) else None
    t2 = float(c2[0] * dt) if len(c2) else None
    # p2 at x=180 is 162 cells from the port edge (x=18) -> the >=100-cell
    # propagation criterion
    travelled = t2 is not None
    speed = (100.0 / (t2 - t1)) if (t1 is not None and t2 is not None
                                    and t2 > t1) else None

    # control: same medium at rest, no stimulus -- must stay quiet
    rd2, _ = build()
    rd2.run(2000)
    quiet = float(rd2.u.max())

    return {'travelled_100_cells': bool(travelled),
            't_p1': t1, 't_p2': t2,
            'n_pulses_p1': int(len(c1)), 'n_pulses_p2': int(len(c2)),
            'speed_cells_per_tu': speed,
            'peak_p2': float(data['p2'].max()),
            'control_max_u_100tu': quiet,
            'spontaneous_firing': bool(quiet > 0.1)}


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'scan'
    if cmd == 'scan':
        scan()
        plot_map()
        cands = pick_candidates()
        print('\nTop candidates closest to literature neighbourhood:')
        for c in cands[:8]:
            print(f"  f={c['f']:.3f} eps={c['eps']:.4f} phi={c['phi']:.3f} "
                  f"{c['class_name']} u*={c['u_star']:.5f} "
                  f"maxRe={c['max_re_eig']:.3f} dist={c['dist_lit']:.3f}")
    elif cmd == 'map':
        plot_map()
    elif cmd == 'propagate':
        with open(CAND_JSON) as fh:
            cands = json.load(fh)['candidates']
        for c in cands:
            print(f"propagating f={c['f']} eps={c['eps']} phi={c['phi']} ...",
                  flush=True)
            res = propagate(c)
            c.update(res)
            print('   ', res, flush=True)
        with open(CAND_JSON, 'w') as fh:
            json.dump({'candidates': cands}, fh, indent=2)
        print(f'Updated {CAND_JSON}')
