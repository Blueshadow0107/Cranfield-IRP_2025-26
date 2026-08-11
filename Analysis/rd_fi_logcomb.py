"""
Log-comb sweep of the f-I onset region (x in ~0.03..0.30, log-spaced).

Motivation: the coarse native sweep (rd_fi_native.py) showed silence below
x~0.05 and a steep ramp from 0.06 to 0.20, but the onset region is
undersampled.  This run maps the firing-rate onset law with 12 log-spaced
levels and measures the onset exponent: excitable-media theory predicts
rate ~ (x - x_c)^beta near threshold (beta ~ 1/2 near a saddle-node).

Same geometry/protocol as rd_fi_native.py (300x40, W=16, sustained DC
clamp u=x / v=u*, probe x=240, 200 t.u. with 50 t.u. discarded).
Outputs: Analysis/figures/rd_fi_logcomb.{json,png}
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rd_core import RDSubstrate

NX, NY = 300, 40
W2 = 8
CY = NY // 2
X_PORT = (0, 18)
X_PROBE = 240
DT = 0.05
TU_DISCARD = 50.0
TU_RUN = 200.0
F, EPS, PHI = 1.4375, 0.05014844822490394, 0.010
U_STAR = 0.0030821
LEVELS = np.geomspace(0.03, 0.30, 12)
OUT_JSON = __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), 'figures', 'rd_fi_logcomb.json')
OUT_PNG = __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), 'figures', 'rd_fi_logcomb.png')


def make_rd():
    wall = np.ones((NX, NY), bool)
    wall[:, CY - W2:CY + W2] = False
    rd = RDSubstrate(nx=NX, ny=NY, dt=DT, kinetics='oregonator',
                     f=F, eps=EPS, clamp_rest=(U_STAR, U_STAR))
    rd.set_phi(PHI)
    rd.set_walls(wall)
    rd.u[~wall] = U_STAR
    rd.v[~wall] = U_STAR
    port = np.zeros((NX, NY), bool)
    port[X_PORT[0]:X_PORT[1], CY - W2:CY + W2] = True
    return rd, wall, port


def run_level(x):
    rd, wall, port = make_rd()
    nsteps = int(TU_RUN / DT)
    ndiscard = int(TU_DISCARD / DT)
    series = []
    for n in range(nsteps):
        rd.u[port] = x
        rd.v[port] = U_STAR
        rd.run(1)
        if n >= ndiscard:
            series.append(float(rd.u[X_PROBE, CY]))
    series = np.array(series)
    idx = np.nonzero((series[:-1] <= 0.5) & (series[1:] > 0.5))[0]
    ncross = len(idx)
    elapsed = (nsteps - ndiscard) * DT
    rate = ncross / elapsed
    isi = np.diff(idx) * DT if ncross > 1 else np.array([])
    return {'level': float(x), 'n_pulses': int(ncross), 'rate': float(rate),
            'rate_rel_noise': float(1 / np.sqrt(ncross)) if ncross else None,
            'isi_mean': float(isi.mean()) if len(isi) else None,
            'isi_cv': float(isi.std() / isi.mean()) if len(isi) else None,
            'peak': float(series.max())}


def main():
    res = [run_level(x) for x in LEVELS]
    for r in res:
        print(f"x={r['level']:.3f} pulses={r['n_pulses']:<4} "
              f"rate={r['rate']:.4f} isi_cv={r['isi_cv']}", flush=True)

    # onset-exponent fit on the firing levels: rate vs (x - x_c), try a
    # small grid of candidate x_c and report the best beta
    firing = [(r['level'], r['rate']) for r in res if r['n_pulses'] >= 8]
    silent_max = max((r['level'] for r in res if r['n_pulses'] == 0),
                     default=None)
    fit = None
    if len(firing) >= 3 and silent_max is not None:
        xs, rs = np.array(firing).T
        best = None
        for xc in np.linspace(silent_max * 0.9, firing[0][0] * 0.98, 40):
            d = xs - xc
            if (d <= 0).any():
                continue
            b, a = np.polyfit(np.log(d), np.log(rs), 1)
            pred = np.exp(a) * d ** b
            err = float(np.sqrt(np.mean((pred - rs) ** 2)))
            if best is None or err < best[2]:
                best = (b, xc, err)
        if best:
            fit = {'beta': best[0], 'x_c': best[1], 'rmse': best[2],
                   'note': 'rate ~ (x - x_c)^beta fit on firing levels'}
            print(f"onset fit: beta={best[0]:.2f}, x_c={best[1]:.3f}, "
                  f"rmse={best[2]:.4f}")

    out = {'levels': res, 'silent_max': silent_max,
           'n_firing': len(firing), 'onset_fit': fit,
           'config': {'geometry': '300x40 W=16, probe x=240',
                      'tu_run': TU_RUN, 'tu_discard': TU_DISCARD}}
    with open(OUT_JSON, 'w') as fh:
        json.dump(out, fh, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    xs = [r['level'] for r in res]
    rs = [r['rate'] for r in res]
    axes[0].plot(xs, rs, 'o-')
    axes[0].set_xlabel('drive level x'); axes[0].set_ylabel('rate (1/t.u.)')
    axes[0].set_title('f-I onset region (log-comb)')
    if fit and len(firing) >= 3:
        fx, fr = np.array(firing).T
        axes[1].loglog(fx - fit['x_c'], fr, 'o')
        d = np.geomspace(min(fx - fit['x_c']), max(fx - fit['x_c']), 50)
        scale = fr[0] / d[0] ** fit['beta']
        axes[1].loglog(d, scale * d ** fit['beta'], '--',
                       label=f"beta={fit['beta']:.2f}")
        axes[1].set_xlabel('x - x_c'); axes[1].set_ylabel('rate')
        axes[1].legend()
        axes[1].set_title('onset exponent')
    else:
        axes[1].plot(xs, rs, 'o-')
        axes[1].set_xlabel('drive level x')
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f'saved {OUT_JSON}, {OUT_PNG}')


if __name__ == '__main__':
    main()
