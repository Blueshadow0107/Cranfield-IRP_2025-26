"""
rd_local_sensitivity.py

Fast one-at-a-time sensitivity of the Oregonator channel-pulse QoIs to the
four scalar parameters around the working point.  Each parameter is varied
independently by +/- 10 % while the others are held at their nominal values.

Outputs:
    Analysis/figures/rd_local_sensitivity/results.json
    Analysis/figures/rd_local_sensitivity/{speed,peak}_sensitivity.png
    Analysis/figures/rd_local_sensitivity/{speed,peak}_sensitivity.pdf
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rd_core import RDSubstrate

NOM = {
    'eps': 0.05014844822490394,
    'f': 1.4375,
    'phi': 0.010,
    'Du': 1.0,
}
FRACTION = 0.10
U_STAR = 0.0030821
THRESH = 0.30
NX, NY = 120, 40
DX = 1.0
DT = 0.05
DURATION_STEPS = 30
TOTAL_STEPS = 500

FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures', 'rd_local_sensitivity')
os.makedirs(FIG_DIR, exist_ok=True)


def _build_rd(eps, f, phi0, Du):
    rd = RDSubstrate(
        nx=NX, ny=NY, dx=DX, dt=DT,
        kinetics='oregonator',
        eps=eps, q=0.002, f=f,
        Du=Du, Dv=0.0,
        clamp_rest=(U_STAR, U_STAR),
    )
    wall = np.zeros((NX, NY), bool)
    wall[:, :10] = True
    wall[:, 30:] = True
    rd.set_walls(wall)
    rd.set_phi(np.full((NX, NY), phi0))
    rd.u[:, :] = U_STAR
    rd.v[:, :] = U_STAR

    port = np.zeros((NX, NY), bool)
    port[5:15, 10:30] = True
    rd.add_port('in', port)

    for name, x in [('p40', 40), ('p100', 100)]:
        mask = np.zeros((NX, NY), bool)
        mask[x:x + 2, 19:21] = True
        rd.add_probe(name, mask)

    rd.fire('in', value=0.8, v_value=0.2, duration=DURATION_STEPS)
    return rd


def evaluate(params):
    eps, f, phi, Du = params
    try:
        rd = _build_rd(eps, f, phi, Du)
        data = rd.run(TOTAL_STEPS)
        t = data['t']
        s40 = data['p40']
        s100 = data['p100']
        idx40 = np.where(s40 > THRESH)[0]
        idx100 = np.where(s100 > THRESH)[0]
        if len(idx40) == 0 or len(idx100) == 0:
            return 0.0, 0.0
        t40 = t[idx40[0]]
        t100 = t[idx100[0]]
        dt = t100 - t40
        if dt <= 0:
            return 0.0, 0.0
        speed = 60.0 / dt
        peak = float(s100.max())
        return speed, peak
    except Exception:
        return 0.0, 0.0


def main():
    names = ['eps', 'f', 'phi', 'Du']
    nom_vec = np.array([NOM[n] for n in names])
    out = {'nominal': NOM, 'names': names}
    nominal_speed, nominal_peak = evaluate(nom_vec)
    out['nominal_speed'] = float(nominal_speed)
    out['nominal_peak'] = float(nominal_peak)

    table = []
    for i, n in enumerate(names):
        base = NOM[n]
        deltas = [-FRACTION * base, 0.0, FRACTION * base]
        speeds, peaks = [], []
        for d in deltas:
            pvec = nom_vec.copy()
            pvec[i] = base + d
            s, pk = evaluate(pvec)
            speeds.append(float(s))
            peaks.append(float(pk))
        # finite-difference sensitivities at +/- 10%
        dlow, dhigh = speeds[0] - speeds[1], speeds[2] - speeds[1]
        p_dlow, p_dhigh = peaks[0] - peaks[1], peaks[2] - peaks[1]
        table.append({
            'param': n,
            'values': [base + d for d in deltas],
            'speeds': speeds,
            'peaks': peaks,
            'speed_delta_low': float(dlow),
            'speed_delta_high': float(dhigh),
            'peak_delta_low': float(p_dlow),
            'peak_delta_high': float(p_dhigh),
            'speed_sensitivity': float((dhigh - dlow) / (2 * FRACTION * base)),
            'peak_sensitivity': float((p_dhigh - p_dlow) / (2 * FRACTION * base)),
        })
        print(f"{n}: speed {speeds[1]:.3f} [{speeds[0]:.3f}, {speeds[2]:.3f}], "
              f"peak {peaks[1]:.3f} [{peaks[0]:.3f}, {peaks[2]:.3f}]")

    out['table'] = table
    with open(os.path.join(FIG_DIR, 'results.json'), 'w') as fh:
        json.dump(out, fh, indent=2)

    # Plot normalised sensitivity of speed and peak
    param_labels = {'eps': r'$\varepsilon$', 'f': '$f$', 'phi': r'$\phi$', 'Du': '$D_u$'}
    params = [param_labels[r['param']] for r in table]
    speed_plus = [r['speed_delta_high'] / out['nominal_speed'] * 100 for r in table]
    speed_minus = [r['speed_delta_low'] / out['nominal_speed'] * 100 for r in table]
    peak_plus = [r['peak_delta_high'] / out['nominal_peak'] * 100 for r in table]
    peak_minus = [r['peak_delta_low'] / out['nominal_peak'] * 100 for r in table]

    x = np.arange(len(params))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.bar(x - width / 2, speed_minus, width, label='-10 %', color='steelblue')
    ax.bar(x + width / 2, speed_plus, width, label='+10 %', color='coral')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(params)
    ax.set_ylabel('relative change in speed (%)')
    ax.set_title('Speed sensitivity')
    ax.legend()

    ax = axes[1]
    ax.bar(x - width / 2, peak_minus, width, label='-10 %', color='steelblue')
    ax.bar(x + width / 2, peak_plus, width, label='+10 %', color='coral')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(params)
    ax.set_ylabel('relative change in peak (%)')
    ax.set_title('Peak-amplitude sensitivity')
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'sensitivity.png'), dpi=200)
    fig.savefig(os.path.join(FIG_DIR, 'sensitivity.pdf'))
    plt.close(fig)
    print(f'[local sensitivity] Saved figures to {FIG_DIR}')


if __name__ == '__main__':
    main()
