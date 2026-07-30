"""
rd_verification.py -- formal convergence and invariants harness for the
de-hacked rd_core solver (goes in the thesis methodology).

Three blocks, for BOTH kinetics (Barkley excitable: a=0.75, b=0.01,
eps=0.02, Du=1, Dv=0; and validated Oregonator Candidate A: f=1.4375,
eps=0.0501, phi=0.010, q=0.002, Du=1, Dv=0.6, fields initialised at the
rest state u*=v*=0.0030821):

  GRID      Free-medium plane-pulse speed at DX = 1.0, 0.5, 0.25 with DT
            scaled to hold the diffusion number D*DT/DX^2 fixed at 0.05.
            Physical domain held at 200 x 40 (i.e. 200x40, 400x80,
            800x160 cells).  Richardson extrapolation from the three
            levels gives the observed order and the estimated asymptotic
            error at the working DX = 1.
  TIMESTEP  Same quantity at fixed DX = 1, DT = 0.05, 0.025, 0.0125.
  INVARIANTS  During a standard walled-channel run: assert no NaN/Inf
            (the rd_core guard would raise), report actual min/max of u
            and v, and the peak adaptive-reaction substep count.

The pulse front at DX=1 is only a few cells thick, so a few-percent DX
sensitivity is expected and is QUANTIFIED, not hidden.

Usage (one case per invocation, for parallel background execution):
    ../.venv/bin/python rd_verification.py --case grid:barkley:1.0
    ../.venv/bin/python rd_verification.py --case dt:oregonator:0.0125
    ../.venv/bin/python rd_verification.py --case invariants:barkley
    ../.venv/bin/python rd_verification.py --collect
Each --case writes a partial JSON to a temp dir; --collect assembles
Analysis/figures/rd_verification.{png,json} and prints the tables.
"""

import argparse
import json
import os
import tempfile

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from rd_core import RDSubstrate

Q = 0.002
U_THRESH = 0.5
DIFFUSION_NUMBER = 0.05          # D*DT/DX^2 held fixed for the grid study
LX, LY = 200.0, 40.0             # physical domain (length units)
X_PORT_END = 18.0                # port slab x in [0, 18)
X_P1, X_P2 = 90.0, 150.0         # probe columns (physical)
FIRE_TU = 1.5                    # port hold duration in time units

OREG_A = dict(f=1.4375, eps=0.05014844822490394, phi=0.010)

KINS = {
    'barkley': dict(kw=dict(kinetics='barkley', eps=0.02, Dv=0.0),
                    phi=0.0, T=50.0,
                    label='Barkley (eps=0.02, Dv=0)'),
    'oregonator': dict(kw=dict(kinetics='oregonator', f=OREG_A['f'],
                               eps=OREG_A['eps']),
                       phi=OREG_A['phi'], T=35.0,
                       label='Oregonator A (eps=0.0501, phi=0.010)'),
}

GRID_DX = [1.0, 0.5, 0.25]
DT_LIST = [0.05, 0.025, 0.0125]

PART_DIR = os.path.join(tempfile.gettempdir(), 'rd_verification_parts')


def rest_u_star(f, phi):
    """Smallest positive root of the u-nullcline with v = u (dv/dt = 0)."""
    def F(u):
        return u - u**2 - (f * u + phi) * (u - Q) / (u + Q)
    ugrid = np.concatenate(([0.0], np.logspace(-7, np.log10(1.2), 4000)))
    vals = F(ugrid)
    for i in range(len(ugrid) - 1):
        if vals[i] > 0.0 and vals[i + 1] <= 0.0:
            return brentq(F, ugrid[i], ugrid[i + 1], xtol=1e-14, rtol=1e-12)
    raise RuntimeError('no rest state found')


def first_crossing_time(t, series, thresh=U_THRESH):
    """First crossing time, linearly interpolated, or None."""
    idx = np.nonzero(series >= thresh)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    if i == 0:
        return float(t[0])
    s0, s1 = series[i - 1], series[i]
    frac = (thresh - s0) / (s1 - s0) if s1 > s0 else 0.0
    return float(t[i - 1] + frac * (t[i] - t[i - 1]))


def speed_run(kin, DX, DT):
    """Free-medium plane-pulse speed in physical length units / t.u."""
    spec = KINS[kin]
    NX, NY = int(round(LX / DX)), int(round(LY / DX))
    rd = RDSubstrate(nx=NX, ny=NY, dx=DX, dt=DT, **spec['kw'])
    if spec['phi']:
        rd.set_phi(spec['phi'])
    rest = 0.0
    if kin == 'oregonator':
        rest = rest_u_star(spec['kw']['f'], spec['phi'])
        rd.u[:] = rest
        rd.v[:] = rest
    cy = NY // 2
    port = np.zeros((NX, NY), bool)
    port[:int(round(X_PORT_END / DX)), :] = True
    rd.add_port('in', port)
    x1, x2 = int(round(X_P1 / DX)), int(round(X_P2 / DX))
    for name, x in (('p1', x1), ('p2', x2)):
        m = np.zeros((NX, NY), bool)
        m[x, cy - 1:cy + 2] = True    # single-column 3-cell strip: no
        rd.add_probe(name, m)         # dilution for a planar front
    duration = int(round(FIRE_TU / DT))
    nsteps = int(round(spec['T'] / DT))
    rd.fire('in', duration=duration)
    data = rd.run(nsteps)
    t1 = first_crossing_time(data['t'], data['p1'])
    t2 = first_crossing_time(data['t'], data['p2'])
    assert t1 is not None and t2 is not None and t2 > t1, \
        f'front did not traverse both probes ({kin}, DX={DX}, DT={DT})'
    dist = (x2 - x1) * DX
    c_phys = dist / (t2 - t1)          # length units per t.u.
    dn = 1.0 * DT / DX**2              # diffusion number (Du = 1)
    return {'kinetics': kin, 'DX': DX, 'DT': DT, 'diffusion_number': dn,
            'grid': [NX, NY], 'nsteps': nsteps,
            't1': t1, 't2': t2, 'dist': dist,
            'c_phys': c_phys, 'c_cells_per_tu': c_phys / DX,
            'peak_p2': float(data['p2'].max())}


def invariant_run(kin):
    """Standard walled-channel run with invariant monitoring."""
    spec = KINS[kin]
    NX, NY, W = (256, 256, 16) if kin == 'barkley' else (300, 48, 16)
    DT = 0.05
    CY = NY // 2
    rd = RDSubstrate(nx=NX, ny=NY, dt=DT, **spec['kw'])
    if spec['phi']:
        rd.set_phi(spec['phi'])
    wall = np.ones((NX, NY), bool)
    wall[:, CY - W // 2:CY + W // 2] = False
    rd.set_walls(wall)
    rest = 0.0
    if kin == 'oregonator':
        rest = rest_u_star(spec['kw']['f'], spec['phi'])
        rd.u[~wall] = rest
        rd.v[~wall] = rest
    port = np.zeros((NX, NY), bool)
    port[0:18, CY - W // 2:CY + W // 2] = True
    rd.add_port('in', port)
    rd.fire('in', duration=30)

    umin, umax = np.inf, -np.inf
    vmin, vmax = np.inf, -np.inf
    max_sub = 0
    nsteps, chunk = 1500, 25
    done = 0
    while done < nsteps:
        rd.run(min(chunk, nsteps - done))
        done += chunk
        assert np.isfinite(rd.u).all() and np.isfinite(rd.v).all(), \
            f'NaN/Inf at step {rd.t} ({kin})'
        umin = min(umin, float(rd.u.min())); umax = max(umax, float(rd.u.max()))
        vmin = min(vmin, float(rd.v.min())); vmax = max(vmax, float(rd.v.max()))
        max_sub = max(max_sub, getattr(rd, '_n_react_sub', 0))
    print(f'  invariants {kin}: u in [{umin:.4f}, {umax:.4f}], '
          f'v in [{vmin:.4f}, {vmax:.4f}], max react substeps {max_sub}',
          flush=True)
    return {'kinetics': kin,
            'geometry': f'{NX}x{NY} walled channel W={W}, dt={DT}, '
                        f'{nsteps} steps',
            'u_min': umin, 'u_max': umax, 'v_min': vmin, 'v_max': vmax,
            'nan_or_inf': False, 'guard_trips': 0,
            'max_react_substeps': int(max_sub),
            'barkley_expected_u_range': [-0.05, 1.05]}


def richardson(vals, ratio=2.0):
    """Richardson analysis of three refinement levels (coarse->fine).

    Returns observed order, extrapolated asymptote, and the estimated
    relative error at the coarsest (working) level.  Non-monotonic data
    is reported as such with the observed spread instead.
    """
    c1, c2, c3 = vals
    d12, d23 = c1 - c2, c2 - c3
    out = {'levels': vals,
           'change_1_2_pct': abs(d12) / abs(c2) * 100,
           'change_2_3_pct': abs(d23) / abs(c3) * 100}
    if d12 * d23 <= 0 or d12 == d23:
        out['monotonic'] = False
        out['spread_pct'] = (max(vals) - min(vals)) / abs(np.mean(vals)) * 100
        out['note'] = ('non-monotonic (temporal and splitting errors '
                       'comparable); spread reported instead of an order')
        out['est_error_at_coarse_pct'] = out['spread_pct']
        return out
    p = np.log2(abs(d12 / d23))
    # c(h) = c_inf + A h^p  ->  c_inf = c3 - (c2 - c3) / (r^p - 1)
    c_inf = c3 - d23 / (ratio**p - 1.0)
    out.update({'monotonic': True, 'observed_order': float(p),
                'c_extrapolated': float(c_inf),
                'est_error_at_coarse_pct': float(
                    abs(c1 - c_inf) / abs(c_inf) * 100)})
    return out


def run_case(spec_str):
    kind, kin, val = (spec_str.split(':') + [None])[:3]
    os.makedirs(PART_DIR, exist_ok=True)
    if kind == 'grid':
        DX = float(val)
        DT = DIFFUSION_NUMBER * DX**2
        res = speed_run(kin, DX, DT)
        tag = f'grid_{kin}_{DX}'
    elif kind == 'dt':
        res = speed_run(kin, 1.0, float(val))
        tag = f'dt_{kin}_{val}'
    elif kind == 'invariants':
        res = invariant_run(kin)
        tag = f'invariants_{kin}'
    else:
        raise ValueError(spec_str)
    p = os.path.join(PART_DIR, f'{tag}.json')
    with open(p, 'w') as fh:
        json.dump(res, fh, indent=2)
    print(f'wrote {p}')
    if kind in ('grid', 'dt'):
        print(f"  {kin} DX={res['DX']} DT={res['DT']}: "
              f"c = {res['c_phys']:.4f} len/t.u. "
              f"({res['c_cells_per_tu']:.4f} cells/t.u.)")


def collect():
    def load(tag):
        with open(os.path.join(PART_DIR, f'{tag}.json')) as fh:
            return json.load(fh)

    results = {'grid': {}, 'dt': {}, 'invariants': {}}
    print('\n================ GRID CONVERGENCE (fixed diffusion number '
          f'{DIFFUSION_NUMBER}) ================')
    for kin in KINS:
        runs = [load(f'grid_{kin}_{dx}') for dx in GRID_DX]
        cs = [r['c_phys'] for r in runs]
        rich = richardson(cs)
        results['grid'][kin] = {'runs': runs, 'richardson': rich}
        print(f"\n{KINS[kin]['label']}: c in length units / t.u.")
        for r in runs:
            print(f"  DX={r['DX']:.2f} DT={r['DT']:.5f} "
                  f"({r['grid'][0]}x{r['grid'][1]}, {r['nsteps']} steps): "
                  f"c = {r['c_phys']:.4f}")
        print(f"  % change DX 1.0->0.5: {rich['change_1_2_pct']:.2f}%   "
              f"0.5->0.25: {rich['change_2_3_pct']:.2f}%")
        if rich.get('monotonic'):
            print(f"  observed order ~ {rich['observed_order']:.2f}, "
                  f"extrapolated c_inf = {rich['c_extrapolated']:.4f}, "
                  f"est. error at DX=1: {rich['est_error_at_coarse_pct']:.2f}%")
        else:
            print(f"  non-monotonic; spread {rich['spread_pct']:.2f}% "
                  f"({rich['note']})")

    print('\n================ TIMESTEP CONVERGENCE (DX = 1) '
          '================')
    for kin in KINS:
        runs = [load(f'dt_{kin}_{dt}') for dt in DT_LIST]
        cs = [r['c_phys'] for r in runs]
        rich = richardson(cs)
        results['dt'][kin] = {'runs': runs, 'richardson': rich}
        print(f"\n{KINS[kin]['label']}: c in length units / t.u. (cells/t.u.)")
        for r in runs:
            print(f"  DT={r['DT']:.5f}: c = {r['c_phys']:.4f}")
        print(f"  % change DT 0.05->0.025: {rich['change_1_2_pct']:.2f}%   "
              f"0.025->0.0125: {rich['change_2_3_pct']:.2f}%")
        if rich.get('monotonic'):
            print(f"  observed order ~ {rich['observed_order']:.2f}, "
                  f"extrapolated c_inf = {rich['c_extrapolated']:.4f}, "
                  f"est. error at DT=0.05: {rich['est_error_at_coarse_pct']:.2f}%")
        else:
            print(f"  non-monotonic; spread {rich['spread_pct']:.2f}% "
                  f"({rich['note']})")

    print('\n================ INVARIANTS (channel run) ================')
    for kin in KINS:
        inv = load(f'invariants_{kin}')
        results['invariants'][kin] = inv
        print(f"  {kin}: u in [{inv['u_min']:.4f}, {inv['u_max']:.4f}], "
              f"v in [{inv['v_min']:.4f}, {inv['v_max']:.4f}], "
              f"NaN/Inf: {inv['nan_or_inf']}, guard trips: "
              f"{inv['guard_trips']}, max react substeps: "
              f"{inv['max_react_substeps']}")

    # --- figure ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    for kin, mk in (('barkley', 'o'), ('oregonator', 's')):
        runs = results['grid'][kin]['runs']
        rich = results['grid'][kin]['richardson']
        ax.plot([r['DX'] for r in runs], [r['c_phys'] for r in runs],
                mk + '-', label=KINS[kin]['label'])
        if rich.get('monotonic'):
            ax.axhline(rich['c_extrapolated'], color=f'C{0 if kin == "barkley" else 1}',
                       ls=':', lw=0.9)
    ax.set_xscale('log', base=2)
    ax.set_xticks(GRID_DX)
    ax.set_xticklabels([str(d) for d in GRID_DX])
    ax.invert_xaxis()
    ax.set_xlabel('grid spacing DX (DT scaled: D*DT/DX^2 = 0.05)')
    ax.set_ylabel('pulse speed (length units / t.u.)')
    ax.set_title('Grid convergence (free-medium pulse speed)')
    ax.legend(fontsize=9)

    ax = axes[1]
    for kin, mk in (('barkley', 'o'), ('oregonator', 's')):
        runs = results['dt'][kin]['runs']
        ax.plot([r['DT'] for r in runs], [r['c_phys'] for r in runs],
                mk + '-', label=KINS[kin]['label'])
    ax.set_xscale('log', base=2)
    ax.set_xticks(DT_LIST)
    ax.set_xticklabels([str(d) for d in DT_LIST])
    ax.invert_xaxis()
    ax.set_xlabel('time step DT (DX = 1)')
    ax.set_ylabel('pulse speed (length units / t.u.)')
    ax.set_title('Timestep convergence')
    ax.legend(fontsize=9)

    fig.suptitle('rd_core verification: convergence of the free-medium '
                 'pulse speed', fontsize=13)
    fig.tight_layout()
    figpath = 'figures/rd_verification.png'
    fig.savefig(figpath, dpi=150)
    plt.close(fig)
    print(f'\nSaved {figpath}')

    jsonpath = 'figures/rd_verification.json'
    with open(jsonpath, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f'Saved {jsonpath}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', action='append', default=[],
                    help='grid:KIN:DX | dt:KIN:DT | invariants:KIN')
    ap.add_argument('--collect', action='store_true')
    args = ap.parse_args()
    for c in args.case:
        run_case(c)
    if args.collect:
        collect()
