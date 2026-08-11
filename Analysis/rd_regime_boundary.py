"""Uniform-phi oscillatory/excitable boundary vs local dark-spot threshold.

Cross-check for the methods chapter: the dark-spot drive response put the
local-patch firing threshold between phi = 0.006 and 0.004 (rd_darkspot).
Here the UNIFORM medium is scanned in phi: rest-initialise at phi, kick a
small disk, and count firings at the kick site over 80 t.u. Sustained
re-firing => oscillatory regime; dies out => excitable.

Output: figures/rd_regime_boundary.json
"""
import json
import numpy as np
from scipy.optimize import brentq
from rd_core import RDSubstrate

NX = NY = 120
DT = 0.05
F, EPS, Q = 1.4375, 0.05014844822490394, 0.002
U_THRESH = 0.5
T_RUN = 80.0

X, Y = np.mgrid[0:NX, 0:NY].astype(float)
KICK = ((X - 30) ** 2 + (Y - 60) ** 2) < 8 ** 2
SITE = ((X - 30) ** 2 + (Y - 60) ** 2) < 9


def rest_u(phi):
    def Fn(u):
        return u - u**2 - (F * u + phi) * (u - Q) / (u + Q)
    ug = np.concatenate(([0.0], np.logspace(-7, np.log10(1.2), 4000)))
    v = Fn(ug)
    for i in range(len(ug) - 1):
        if v[i] > 0.0 and v[i + 1] <= 0.0:
            return brentq(Fn, ug[i], ug[i + 1], xtol=1e-14, rtol=1e-12)
    return 0.0


out = {}
for phi in (0.002, 0.004, 0.006, 0.008, 0.010, 0.012):
    us = rest_u(phi)
    rd = RDSubstrate(nx=NX, ny=NY, dt=DT, kinetics='oregonator',
                     f=F, eps=EPS, clamp_rest=(us, us))
    rd.set_phi(np.full((NX, NY), phi))
    rd.u[:] = us
    rd.v[:] = us
    rd.u[KICK] = 0.6           # supra-threshold kick
    series = np.empty(int(T_RUN / DT))
    for n in range(len(series)):
        rd.run(1)
        series[n] = rd.u[SITE].max()
    above = series >= U_THRESH
    crossings = int(np.sum(~above[:-1] & above[1:]))
    quiet_tail = bool(series[-int(10 / DT):].max() < U_THRESH)
    out[str(phi)] = {'rest_u': us, 'firings_at_site_80tu': crossings,
                     'quiet_final_10tu': quiet_tail,
                     'regime': 'oscillatory' if crossings >= 3
                               and not quiet_tail else 'excitable/dying'}
    print(f'phi={phi:.3f} rest={us:.5f} firings={crossings} '
          f'quiet_tail={quiet_tail} -> {out[str(phi)]["regime"]}',
          flush=True)

with open('figures/rd_regime_boundary.json', 'w') as fh:
    json.dump(out, fh, indent=2)
print('[done] figures/rd_regime_boundary.json', flush=True)
