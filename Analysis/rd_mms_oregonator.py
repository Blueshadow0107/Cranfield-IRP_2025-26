"""
MMS convergence harness for rd_core (method of manufactured solutions).

Verifies the spatial and temporal discretization orders of the de-hacked
RDSubstrate solver against the exact manufactured fields from
rd_mms_sources.py (sources act through the FULL operators: reaction +
divergence-form tensor diffusion, evaluated via the set_source hook --
first-order-in-time explicit source, see rd_core.set_source docstring).

Cases:
  (a) barkley_iso      -- Barkley, isotropic D = 1
  (b) barkley_tilted   -- Barkley, tilted tensor Dxx=Dyy=2.5, Dxy=1.5
                          (key test of the cross-derivative path)
  (c) oregonator_A     -- Oregonator Candidate A, isotropic D = 1,
                          Dv = 0.6, phi = 0.010

Protocols (domain 64x64 physical, final time t = 20 = one period T):
  spatial : DX = 1, 0.5, 0.25 with DT = 0.05 * DX^2  -> expect order ~2
            (diffusion discretization; the DT scaling keeps the O(dt)
            splitting/source error proportional to DX^2 as well)
  temporal: DX = 0.5 fixed, DT = 0.0125, 0.00625, 0.003125 -> expect ~1
            (explicit diffusion + Godunov splitting + explicit source)

Errors (L2 = RMS, Linf = max) are measured against the exact
manufactured fields at the final time.  Observed orders are
p = log2(E_coarse / E_fine) between successive refinements; the
reported order is the fine-pair value (0.5->0.25 spatial,
0.00625->0.003125 temporal).

Usage:
    ../.venv/bin/python rd_mms.py [case_name|all]

Outputs: Analysis/figures/rd_mms_oregonator.{png,json}
(Oregonator-only copy of rd_mms.py, 2026-08-03: Barkley
dropped from the thesis; slope references anchored at the
finest data point instead of an arbitrary intercept.)
"""

import json
import sys

import numpy as np
import matplotlib.pyplot as plt

from rd_core import RDSubstrate
from rd_mms_sources import get_case, LX, T

T_END = T                       # 20 t.u. = one full period
CASE_NAMES = ['oregonator_A', 'oregonator_tilted']

SPATIAL = [(1.0, 0.05), (0.5, 0.0125), (0.25, 0.003125)]   # (DX, DT)
TEMPORAL_DX = 0.5
TEMPORAL_DTS = [0.0125, 0.00625, 0.003125]

EXPECTED = {'spatial': 2.0, 'temporal': 1.0}


def run_mms(case, dx, dt):
    """One MMS run to t = T_END; return error norms for u and v."""
    n = int(round(LX / dx))
    rd = RDSubstrate(nx=n, ny=n, dx=dx, dt=dt, **case['solver_kw'])
    if case['phi']:
        rd.set_phi(case['phi'])
    rd.set_diffusion_tensor(*case['tensor'])
    X, Y = rd._coords()
    rd.u[:] = case['u_hat'](X, Y, 0.0)
    rd.v[:] = case['v_hat'](X, Y, 0.0)
    rd.set_source(case['Su'], case['Sv'])
    nsteps = int(round(T_END / dt))
    rd.run(nsteps)
    te = nsteps * dt
    eu = rd.u - case['u_hat'](X, Y, te)
    ev = rd.v - case['v_hat'](X, Y, te)
    return {
        'dx': dx, 'dt': dt, 'n': n, 'nsteps': nsteps,
        'u_L2': float(np.sqrt(np.mean(eu**2))),
        'u_Linf': float(np.abs(eu).max()),
        'v_L2': float(np.sqrt(np.mean(ev**2))),
        'v_Linf': float(np.abs(ev).max()),
    }


def orders(rows, key):
    """Observed orders between successive rows for error column `key`."""
    out = []
    for r0, r1 in zip(rows[:-1], rows[1:]):
        refine = (r0['dx'] / r1['dx']) if r0['dx'] != r1['dx'] \
            else (r0['dt'] / r1['dt'])
        out.append(float(np.log(r0[key] / r1[key]) / np.log(refine)))
    return out


def run_case(name):
    case = get_case(name)
    print(f'=== MMS case {name}: {case["kinetics"]}, tensor={case["tensor"]}, '
          f'Dv={case["Dv"]} ===')
    res = {'kinetics': case['kinetics'], 'tensor': case['tensor'],
           'Dv': case['Dv'], 'spatial': [], 'temporal': []}

    print(' spatial sweep (DT = 0.05*DX^2):')
    for dx, dt in SPATIAL:
        r = run_mms(case, dx, dt)
        res['spatial'].append(r)
        print(f'  DX={dx:<5} DT={dt:<8} n={r["n"]:>3} steps={r["nsteps"]:>5} '
              f'u_L2={r["u_L2"]:.3e} u_Linf={r["u_Linf"]:.3e} '
              f'v_L2={r["v_L2"]:.3e}', flush=True)
    print(' temporal sweep (DX = 0.5):')
    for dt in TEMPORAL_DTS:
        r = run_mms(case, TEMPORAL_DX, dt)
        res['temporal'].append(r)
        print(f'  DX={TEMPORAL_DX:<5} DT={dt:<8} n={r["n"]:>3} '
              f'steps={r["nsteps"]:>5} u_L2={r["u_L2"]:.3e} '
              f'u_Linf={r["u_Linf"]:.3e} v_L2={r["v_L2"]:.3e}', flush=True)

    for study in ('spatial', 'temporal'):
        rows = res[study]
        for fld in ('u', 'v'):
            for norm in ('L2', 'Linf'):
                res[f'order_{study}_{fld}_{norm}'] = orders(
                    rows, f'{fld}_{norm}')
    p_sp = res['order_spatial_u_L2'][-1]
    p_tm = res['order_temporal_u_L2'][-1]
    print(f'  observed orders (u, L2): spatial {p_sp:.2f} '
          f'(expect ~2), temporal {p_tm:.2f} (expect ~1)', flush=True)
    return res


def make_figure(results):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    for name, res in results.items():
        rows = res['spatial']
        ax.loglog([r['dx'] for r in rows], [r['u_L2'] for r in rows], 'o-',
                  label=f"{name} (p={res['order_spatial_u_L2'][-1]:.2f})")
    dxs = np.array([0.25, 1.0])
    ref = next(iter(results.values()))['spatial']
    x0, y0 = ref[-1]['dx'], ref[-1]['u_L2']
    ax.loglog(dxs, y0 * (dxs / x0)**2, 'k--', lw=0.8,
              label='slope 2 (expected, anchored)')
    ax.set_xlabel('DX'); ax.set_ylabel('u L2 error at t=20')
    ax.set_title('Spatial convergence')
    ax.legend(fontsize=8)

    ax = axes[1]
    for name, res in results.items():
        rows = res['temporal']
        ax.loglog([r['dt'] for r in rows], [r['u_L2'] for r in rows], 's-',
                  label=f"{name} (p={res['order_temporal_u_L2'][-1]:.2f})")
    dts = np.array(TEMPORAL_DTS)
    ref = next(iter(results.values()))['temporal']
    x0, y0 = ref[-1]['dt'], ref[-1]['u_L2']
    ax.loglog(dts, y0 * (dts / x0), 'k--', lw=0.8,
              label='slope 1 (expected, anchored)')
    ax.set_xlabel('DT'); ax.set_ylabel('u L2 error at t=20')
    ax.set_title('Temporal convergence')
    ax.legend(fontsize=8)

    fig.suptitle('MMS verification of rd_core (manufactured sources '
                 'through full operators)', fontsize=13)
    fig.tight_layout()
    figpath = 'figures/rd_mms_oregonator.png'
    fig.savefig(figpath, dpi=150)
    plt.close(fig)
    print(f'Saved {figpath}')


def print_order_table(results):
    print('\nObserved-vs-expected order table (u field; fine pair):')
    print(f'{"case":<18}{"study":<10}{"L2 order":>10}{"Linf order":>12}'
          f'{"expected":>10}')
    for name, res in results.items():
        for study in ('spatial', 'temporal'):
            p2 = res[f'order_{study}_u_L2'][-1]
            pi = res[f'order_{study}_u_Linf'][-1]
            print(f'{name:<18}{study:<10}{p2:>10.2f}{pi:>12.2f}'
                  f'{EXPECTED[study]:>10.1f}')


def main(which):
    names = CASE_NAMES if which == 'all' else [which]
    results = {}
    for name in names:
        results[name] = run_case(name)
    print_order_table(results)
    make_figure(results)
    jsonpath = 'figures/rd_mms_oregonator.json'
    with open(jsonpath, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f'Saved {jsonpath}')


if __name__ == '__main__':
    arg = sys.argv[1] if len(sys.argv) > 1 else 'all'
    assert arg in CASE_NAMES + ['all'], arg
    main(arg)
