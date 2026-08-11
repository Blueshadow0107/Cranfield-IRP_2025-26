"""Flash calibration: pulses emitted + dim-to-fire latency vs T_FLASH."""
import json
import numpy as np
from rd_core import RDSubstrate

NX, NY = 200, 120
DT = 0.05
PHI0, DARK = 0.010, 0.002
U_STAR = 0.0030821
U_THRESH = 0.5
X, Y = np.mgrid[0:NX, 0:NY].astype(float)
SPOT = ((X - 40) ** 2 + (Y - 60) ** 2) < 12 ** 2
PROBE = ((X - 110) ** 2 + (Y - 60) ** 2) < 9

out = {}
for t_flash in (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0):
    rd = RDSubstrate(nx=NX, ny=NY, dt=DT, kinetics='oregonator',
                     f=1.4375, eps=0.05014844822490394,
                     clamp_rest=(U_STAR, U_STAR))
    base = np.full((NX, NY), PHI0)
    rd.set_phi(base)
    rd.u[:] = U_STAR
    rd.v[:] = U_STAR
    dark = base.copy()
    dark[SPOT] = DARK
    rd.set_phi(dark)
    rd.run(int(t_flash / DT))
    rd.set_phi(base)
    series = np.empty(int(40 / DT))
    for n in range(len(series)):
        rd.run(1)
        series[n] = rd.u[PROBE].max()
    above = series >= U_THRESH
    crossings = int(np.sum(~above[:-1] & above[1:]))
    first = float(np.argmax(above) * DT) if above.any() else None
    out[str(t_flash)] = {'pulses': crossings, 'first_arrival_tu': first,
                         'probe_peak': float(series.max())}
    print(f'T_FLASH={t_flash:5.1f}  pulses={crossings}  '
          f'first={first}  peak={series.max():.3f}', flush=True)

with open('figures/rd_flash_calib.json', 'w') as fh:
    json.dump(out, fh, indent=2)
print('[done] figures/rd_flash_calib.json', flush=True)
