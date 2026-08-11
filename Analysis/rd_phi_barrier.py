"""Blocking diagnostic: can a phi patch stop a propagating pulse?

Motivation: the demux DE campaign sat at loss=2.5 (baseline) for ~490
evals -> phi in [0.010, 0.025] may not block propagation at all, making
routing unlearnable. Here a pulse is fired down a channel through a
barrier band of elevated phi; transmission past the barrier vs barrier
phi gives the blocking threshold and hence the honest PHI_MAX.

Geometry: 220x60 free medium, dark-slab source at x[0,18) (T_FLASH=3,
calibrated single pulse), barrier band x[100,120) at phi_b spanning the
full height, probe strip x=180. Oregonator A, phi0=0.010 elsewhere.

Output: figures/rd_phi_barrier.json
"""
import json
import numpy as np
from rd_core import RDSubstrate

NX, NY = 220, 60
DT = 0.05
PHI0, DARK = 0.010, 0.002
U_STAR = 0.0030821
U_THRESH = 0.5
T_FLASH = 3.0
T_RUN = 50.0

X, Y = np.mgrid[0:NX, 0:NY].astype(float)
SRC = np.zeros((NX, NY), bool)
SRC[0:18, :] = True
PROBE = np.zeros((NX, NY), bool)
PROBE[180, 28:32] = True
BARRIER = np.zeros((NX, NY), bool)
BARRIER[100:120, :] = True

out = {}
for phi_b in (0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050, 0.060):
    rd = RDSubstrate(nx=NX, ny=NY, dt=DT, kinetics='oregonator',
                     f=1.4375, eps=0.05014844822490394,
                     clamp_rest=(U_STAR, U_STAR))
    base = np.full((NX, NY), PHI0)
    base[BARRIER] = phi_b
    rd.set_phi(base)
    rd.u[:] = U_STAR
    rd.v[:] = U_STAR
    dark = base.copy()
    dark[SRC] = DARK
    rd.set_phi(dark)
    rd.run(int(T_FLASH / DT))
    rd.set_phi(base)
    series = np.empty(int(T_RUN / DT))
    for n in range(len(series)):
        rd.run(1)
        series[n] = rd.u[PROBE].max()
    peak = float(series.max())
    transmitted = peak >= U_THRESH
    arr = float(np.argmax(series >= U_THRESH) * DT) if transmitted else None
    out[str(phi_b)] = {'peak': peak, 'transmitted': transmitted,
                       'arrival_tu': arr}
    print(f'phi_b={phi_b:.3f}  peak={peak:.3f}  '
          f'transmitted={transmitted}  arrival={arr}', flush=True)

with open('figures/rd_phi_barrier.json', 'w') as fh:
    json.dump(out, fh, indent=2)
print('[done] figures/rd_phi_barrier.json', flush=True)
