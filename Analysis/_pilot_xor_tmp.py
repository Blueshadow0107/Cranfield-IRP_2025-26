"""TEMPORARY pilot for the XOR phi-training campaign (delete after use).

Measures, at uniform phi = phi0 on the Test-3 final2 Oregonator geometry:
  - wall-clock per 1000-step sim (and for a 4-case concurrent batch),
  - A-alone and B-alone arrival times at the output probe,
  - full-trace peaks for all four input cases (window choice for XOR).
"""
import time
from multiprocessing import Pool

import numpy as np
from scipy.optimize import brentq

from rd_core import RDSubstrate

Q = 0.002
DT = 0.05
W = 20
W2 = W // 2
TJ = 96
X_PROBE = 240
NX, NY = 256, 200
CY = NY // 2
F = 1.4375
EPS = 0.05014844822490394
PHI0 = 0.010


def rest_u_star(f, phi):
    def Fn(u):
        return u - u**2 - (f * u + phi) * (u - Q) / (u + Q)
    g = np.concatenate(([0.0], np.logspace(-7, np.log10(1.2), 4000)))
    v = Fn(g)
    for i in range(len(g) - 1):
        if v[i] > 0 and v[i + 1] <= 0:
            return brentq(Fn, g[i], g[i + 1], xtol=1e-14, rtol=1e-12)
    raise RuntimeError('no rest state')


US = rest_u_star(F, PHI0)


def make():
    wall = np.ones((NX, NY), bool)
    wall[:, CY - W2:CY + W2] = False
    wall[TJ - W2:TJ + W2, CY:172] = False
    rd = RDSubstrate(nx=NX, ny=NY, dt=DT, kinetics='oregonator', f=F,
                     eps=EPS, clamp_rest=(US, US))
    rd.set_phi(PHI0)
    rd.set_walls(wall)
    rd.u[~wall] = US
    rd.v[~wall] = US
    pA = np.zeros((NX, NY), bool)
    pA[25:43, CY - W2:CY + W2] = True
    pB = np.zeros((NX, NY), bool)
    pB[TJ - W2:TJ + W2, 140:158] = True
    rd.add_port('A', pA)
    rd.add_port('B', pB)
    pr = np.zeros((NX, NY), bool)
    pr[X_PROBE, CY - 1:CY + 2] = True
    rd.add_probe('out', pr & ~wall)
    return rd


def case(args):
    fa, fb, tag = args
    t0 = time.time()
    rd = make()
    if fa:
        rd.pulse_train('A', [0], duration=30)
    if fb:
        rd.pulse_train('B', [0], duration=30)
    d = rd.run(1000)
    return tag, time.time() - t0, d['t'], d['out']


def main():
    print(f'u* = {US:.7f}')
    t0 = time.time()
    with Pool(4) as p:
        res = p.map(case, [(True, False, '10'), (False, True, '01'),
                           (True, True, '11'), (False, False, '00')])
    print(f'4-case concurrent batch wall-clock: {time.time() - t0:.1f} s')
    for tag, dt_, t, s in res:
        pk = float(s.max())
        tpk = float(t[np.argmax(s)])
        idx = np.nonzero(s >= 0.5)[0]
        tc = float(t[idx[0]]) if len(idx) else None
        # peak times above rest, to see all arrivals in the trace
        print(f'{tag}: wall={dt_:.1f}s  full-trace peak={pk:.4f} at '
              f't={tpk:.2f}  first 0.5-crossing={tc}')
        # crude arrival listing: local maxima above 0.1
        above = s > 0.1
        edges = np.nonzero((~above[:-1]) & above[1:])[0]
        print(f'    times crossing 0.1 upward: '
              f'{[round(float(t[i]), 2) for i in edges]}')


if __name__ == '__main__':
    main()
