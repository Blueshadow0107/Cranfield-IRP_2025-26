"""
Confirm/deny the f-I dip at high drive (rd_fi_native showed rate falling
from 0.233 at x=0.2 to 0.173 at x=0.45, but with only ~30 pulses the
estimate noise was ~1.5 sigma of the dip depth).

This run: three levels (x = 0.20 peak, 0.45 dip, 0.65 past-dip), long
runs (2000 t.u., first 200 discarded) giving ~300 pulses -> noise ~0.006.
Also records the inhibitor level v in the channel cells just downstream
of the port (the 'overdrive halo' diagnostic: if sustained strong drive
keeps v elevated, the effective threshold rises and the rate drops).

Outputs: Analysis/figures/rd_fi_dip_confirm.json + printed table.
"""

import json
import numpy as np

from rd_core import RDSubstrate

NX, NY = 300, 40
W2 = 8
CY = NY // 2
X_PORT = (0, 18)
X_PROBE = 240
DT = 0.05
TU_DISCARD = 200.0
TU_RUN = 2000.0
F, EPS, Q, PHI = 1.4375, 0.05014844822490394, 0.002, 0.010
U_STAR = 0.0030821
LEVELS = [0.20, 0.45, 0.65]
OUT = __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), 'figures', 'rd_fi_dip_confirm.json')


def make_rd():
    wall = np.ones((NX, NY), bool)
    wall[:, CY - W2:CY + W2] = False
    rd = RDSubstrate(nx=NX, ny=NY, dt=DT, kinetics='oregonator',
                     f=F, eps=EPS, clamp_rest=(U_STAR, U_STAR))
    rd.set_phi(PHI)
    rd.set_walls(wall)
    rd.u[~wall] = U_STAR
    rd.v[~wall] = U_STAR
    probe = np.zeros((NX, NY), bool)
    probe[X_PROBE, CY - 1:CY + 2] = True
    rd.add_probe('out', probe & ~wall)
    port = np.zeros((NX, NY), bool)
    port[X_PORT[0]:X_PORT[1], CY - W2:CY + W2] = True
    return rd, wall, port


def crossings(series, thresh=0.5):
    above = series > thresh
    return int(np.sum((~above[:-1]) & above[1:]))


def run_level_full(x):
    rd, wall, port = make_rd()
    nsteps = int(TU_RUN / DT)
    ndiscard = int(TU_DISCARD / DT)
    probe_series = []
    v_halo = []
    halo = np.zeros((NX, NY), bool)
    halo[20:40, CY - W2:CY + W2] = True
    for n in range(nsteps):
        rd.u[port] = x
        rd.v[port] = U_STAR
        rd.run(1)
        if n >= ndiscard:
            probe_series.append(float(rd.u[X_PROBE, CY]))
            if n % 200 == 0:
                v_halo.append(float(rd.v[halo].mean()))
    probe_series = np.array(probe_series)
    ncross = crossings(probe_series)
    elapsed = (nsteps - ndiscard) * DT
    rate = ncross / elapsed
    noise = 1.0 / np.sqrt(max(ncross, 1)) * rate if ncross else 0.0
    # ISI stats from crossing indices
    idx = np.nonzero((probe_series[:-1] <= 0.5) & (probe_series[1:] > 0.5))[0]
    isi = np.diff(idx) * DT if len(idx) > 1 else np.array([])
    return {
        'level': x,
        'n_pulses': int(ncross),
        'rate': rate,
        'rate_noise': float(noise),
        'isi_mean': float(isi.mean()) if len(isi) else None,
        'isi_cv': float(isi.std() / isi.mean()) if len(isi) else None,
        'v_halo_mean': float(np.mean(v_halo)) if v_halo else None,
        'v_halo_max': float(np.max(v_halo)) if v_halo else None,
    }


if __name__ == '__main__':
    out = {'config': {'levels': LEVELS, 'tu_run': TU_RUN,
                      'tu_discard': TU_DISCARD, 'width': 2 * W2,
                      'v_halo_region': 'channel cells x in [20,40]'},
           'results': []}
    for x in LEVELS:
        r = run_level_full(x)
        out['results'].append(r)
        print(f"x={x:<5} pulses={r['n_pulses']:<5} rate={r['rate']:.4f}"
              f" +/- {r['rate_noise']:.4f}  isi_cv={r['isi_cv']}  "
              f"v_halo={r['v_halo_mean']:.4f} (max {r['v_halo_max']:.4f})",
              flush=True)
    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(f'saved {OUT}')
