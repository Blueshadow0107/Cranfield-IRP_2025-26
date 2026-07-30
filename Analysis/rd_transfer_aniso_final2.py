"""
TEST 4a (FINAL2, clamp-rest fix) -- sqrt(r) ellipse check, both kinetics.

Adapted copy of rd_transfer_aniso_final.py, extended to the excitable
Oregonator Candidate A in addition to Barkley.  Uniform anisotropic
medium, D_perp = 1, D_par = r (fast axis = x), r in {1, 2, 4} at
dt = 0.05.  A central disk pulse is seeded; the wavefront extent along
x and y is tracked over time giving axis speeds c_par, c_perp.
Check c_par/c_perp ~ sqrt(r).

No ports are used (direct field seeding), so the clamp-rest fix does not
change the protocol; for the Oregonator the medium is initialised at
(u*, u*) and a no-stimulus control verifies quiet.  The disk seed is the
reference-script stimulus (u=0.8, v=0.2 inside r=25).

Outputs: Analysis/figures/rd_transfer_aniso_final2.{png,json}
"""

import json
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from rd_core import RDSubstrate

NX, NY = 256, 256
CX, CY = NX // 2, NY // 2
U_THRESH = 0.5
Q = 0.002

OREG_A = dict(f=1.4375, eps=0.05014844822490394, phi=0.010)

KINETICS = {
    'barkley': dict(kw=dict(kinetics='barkley', eps=0.02, Dv=0.0),
                    phi=0.0, label='Barkley (a=0.75, b=0.01, eps=0.02, Dv=0)'),
    'oregonator': dict(kw=dict(kinetics='oregonator', f=OREG_A['f'],
                               eps=OREG_A['eps']),
                       phi=OREG_A['phi'],
                       label='Oregonator A (f=1.4375, eps=0.0501, phi=0.010)'),
}

CASES = [(1, 0.05, 900), (2, 0.05, 700), (4, 0.05, 500)]


def rest_u_star(f, phi):
    def F(u):
        return u - u**2 - (f * u + phi) * (u - Q) / (u + Q)
    ugrid = np.concatenate(([0.0], np.logspace(-7, np.log10(1.2), 4000)))
    vals = F(ugrid)
    for i in range(len(ugrid) - 1):
        if vals[i] > 0.0 and vals[i + 1] <= 0.0:
            return brentq(F, ugrid[i], ugrid[i + 1], xtol=1e-14, rtol=1e-12)
    raise RuntimeError('no rest state found')


def rest_state(kin):
    if kin == 'oregonator':
        us = rest_u_star(KINETICS[kin]['kw']['f'], KINETICS[kin]['phi'])
        return us, us
    return 0.0, 0.0


def disk_mask(cx, cy, r):
    X, Y = np.mgrid[0:NX, 0:NY]
    return (X - cx)**2 + (Y - cy)**2 <= r**2


def tensor_components(r, theta_deg):
    """Tensor with eigenvalues (r, 1), fast axis at theta to the x-axis."""
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    return (r * c**2 + 1.0 * s**2, r * s**2 + 1.0 * c**2,
            (r - 1.0) * s * c)


def front_extents(u, thresh=U_THRESH):
    """Outermost front distance from centre along +x and +y."""
    row = u[:, CY]
    col = u[CX, :]
    xs = np.nonzero(row > thresh)[0]
    ys = np.nonzero(col > thresh)[0]
    ex = (xs.max() - CX) if len(xs) else 0
    ey = (ys.max() - CY) if len(ys) else 0
    return ex, ey


def fit_speed(times, extents, lo=30, hi=100):
    """Linear fit of extent vs time over the window [lo, hi] cells."""
    sel = [(t, e) for t, e in zip(times, extents) if lo <= e <= hi]
    if len(sel) < 5:
        return None
    t = np.array([s[0] for s in sel])
    e = np.array([s[1] for s in sel])
    slope, _ = np.polyfit(t, e, 1)
    return float(slope)  # cells per time unit


def run_4a(kin, r, dt, nsteps, snap_times=()):
    spec = KINETICS[kin]
    u_rest, v_rest = rest_state(kin)
    rd = RDSubstrate(nx=NX, ny=NY, dt=dt, clamp_rest=(u_rest, v_rest),
                     **spec['kw'])
    if spec['phi']:
        rd.set_phi(spec['phi'])
    Dxx, Dyy, Dxy = tensor_components(r, 0.0)   # fast axis = x
    rd.set_diffusion_tensor(Dxx, Dyy, Dxy)
    rd.u[:] = u_rest
    rd.v[:] = v_rest
    seed = disk_mask(CX, CY, 25)
    rd.u[seed] = 0.8
    rd.v[seed] = 0.2

    times, exs, eys = [], [], []
    snaps = {}
    chunk = 10
    done = 0
    while done < nsteps:
        rd.run(min(chunk, nsteps - done))
        done += chunk
        t = rd.t * dt
        ex, ey = front_extents(rd.u)
        times.append(t); exs.append(ex); eys.append(ey)
        for ts in snap_times:
            if abs(t - ts) < 0.5 * chunk * dt and ts not in snaps:
                snaps[ts] = rd.u.copy()
    c_par = fit_speed(times, exs)
    c_perp = fit_speed(times, eys)
    return c_par, c_perp, snaps, (times, exs, eys)


def control_run(kin):
    """No-stimulus anisotropic control (r=4): medium must stay at rest."""
    spec = KINETICS[kin]
    u_rest, v_rest = rest_state(kin)
    rd = RDSubstrate(nx=NX, ny=NY, dt=0.05, clamp_rest=(u_rest, v_rest),
                     **spec['kw'])
    if spec['phi']:
        rd.set_phi(spec['phi'])
    Dxx, Dyy, Dxy = tensor_components(4, 0.0)
    rd.set_diffusion_tensor(Dxx, Dyy, Dxy)
    rd.u[:] = u_rest
    rd.v[:] = v_rest
    rd.run(2000)
    dev = float(np.abs(rd.u - u_rest).max())
    quiet = dev < 0.01
    print(f'  control (r=4, no stimulus, 100 t.u.): max |u-u*| = {dev:.3e} '
          f'-> {"QUIET" if quiet else "SPONTANEOUS FIRING!"}')
    return {'rest_u': u_rest, 'max_dev_100tu': dev, 'quiet': bool(quiet)}


def run_kinetics(kin):
    print(f'--- TEST 4a FINAL2 ({kin}): axis speeds vs anisotropy ---')
    out = {'kinetics_label': KINETICS[kin]['label'],
           'control': control_run(kin), 'cases': {}}
    tracks = {}
    for r, dt, nsteps in CASES:
        c_par, c_perp, snaps, track = run_4a(kin, r, dt, nsteps,
                                             snap_times=(20, 40, 60))
        ratio = c_par / c_perp
        pred = np.sqrt(r)
        dev = abs(ratio - pred) / pred * 100
        out['cases'][str(r)] = {
            'dt': dt, 'c_par': c_par, 'c_perp': c_perp,
            'ratio': ratio, 'sqrt_r': float(pred),
            'deviation_pct': float(dev)}
        tracks[r] = track
        print(f'  r={r} (dt={dt}): c_par={c_par:.3f}, c_perp={c_perp:.3f} '
              f'cells/t.u.; ratio={ratio:.3f} vs sqrt(r)={pred:.3f} '
              f'({dev:.2f}% off)', flush=True)
    return out, tracks


def main(which):
    kins = list(KINETICS) if which == 'all' else [which]
    parts, all_tracks = {}, {}
    for kin in kins:
        parts[kin], all_tracks[kin] = run_kinetics(kin)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for row, kin in enumerate(kins):
        out = parts[kin]
        ax = axes[row, 0]
        times, exs, eys = all_tracks[kin][4]
        ax.plot(times, exs, label='x extent (fast)')
        ax.plot(times, eys, label='y extent (slow)')
        ax.set_xlabel('t'); ax.set_ylabel('front extent (cells)')
        ax.set_title(f'{kin}: r = 4 front expansion')
        ax.legend()
        ax = axes[row, 1]
        rs = sorted(float(k) for k in out['cases'])
        ratios = [out['cases'][str(int(r))]['ratio'] for r in rs]
        ax.plot(rs, ratios, 'o', color='tab:red', ms=8,
                label='measured c_par/c_perp')
        rf = np.linspace(1, 4, 100)
        ax.plot(rf, np.sqrt(rf), 'k--', label='sqrt(r)')
        for r, ratio in zip(rs, ratios):
            c = out['cases'][str(int(r))]
            ax.annotate(f"{c['deviation_pct']:.1f}%", (r, ratio),
                        textcoords='offset points', xytext=(8, -12),
                        fontsize=9)
        ax.set_xlabel('anisotropy ratio r = D_par/D_perp')
        ax.set_ylabel('c_par / c_perp')
        ax.set_title(f'{kin}: sqrt(r) check')
        ax.legend()
    fig.suptitle('TEST 4a FINAL2 (clamp-rest fix): anisotropic target '
                 'wave, both kinetics', fontsize=13)
    fig.tight_layout()
    figpath = 'figures/rd_transfer_aniso_final2.png'
    fig.savefig(figpath, dpi=150)
    plt.close(fig)
    print(f'Saved {figpath}')

    jsonpath = 'figures/rd_transfer_aniso_final2.json'
    with open(jsonpath, 'w') as fh:
        json.dump(parts, fh, indent=2)
    print(f'Saved {jsonpath}')


if __name__ == '__main__':
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else 'all'
    assert arg in ('barkley', 'oregonator', 'all'), arg
    main(arg)
