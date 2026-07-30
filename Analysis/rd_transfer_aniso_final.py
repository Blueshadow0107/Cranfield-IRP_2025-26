"""
TEST 4a (FINAL, de-hacked rd_core) -- sqrt(r) ellipse check.

Adapted copy of the 4a part of rd_transfer_aniso.py for the post-de-hack
verification numbers (the original script and outputs are untouched).
Only part 4a is re-run, on the excitable Barkley model
(a=0.75, b=0.01, eps=0.02, Du tensor, Dv=0):

    Uniform anisotropic medium, D_perp = 1, D_par = r (fast axis = x),
    r in {1, 2, 4} at dt = 0.05 plus r = 8 at dt = 0.02.  A central disk
    pulse is seeded; the wavefront extent along x and y is tracked over
    time giving axis speeds c_par, c_perp.  Check c_par/c_perp ~ sqrt(r).

Measurement (unchanged): front extent = outermost u > 0.5 crossing from
the centre along each axis; speed = least-squares slope of extent vs time
over the 30..100 cell window.

Outputs: Analysis/figures/rd_transfer_aniso_final.{png,json}
"""

import json
import numpy as np
import matplotlib.pyplot as plt

from rd_core import RDSubstrate

NX, NY = 256, 256
CX, CY = NX // 2, NY // 2
U_THRESH = 0.5
KIN = dict(kinetics='barkley', eps=0.02, Dv=0.0)

CASES = [(1, 0.05, 900), (2, 0.05, 700), (4, 0.05, 500), (8, 0.02, 900)]


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


def run_4a(r, dt, nsteps, snap_times=()):
    rd = RDSubstrate(nx=NX, ny=NY, dt=dt, **KIN)
    Dxx, Dyy, Dxy = tensor_components(r, 0.0)   # fast axis = x
    rd.set_diffusion_tensor(Dxx, Dyy, Dxy)
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


def main():
    print('--- TEST 4a FINAL (de-hacked): axis speeds vs anisotropy ratio '
          '(barkley) ---')
    out = {'kinetics': 'barkley (a=0.75, b=0.01, eps=0.02, Dv=0), '
                       'de-hacked rd_core',
           'cases': {}}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for r, dt, nsteps in CASES:
        c_par, c_perp, snaps, track = run_4a(r, dt, nsteps,
                                             snap_times=(20, 40, 60))
        ratio = c_par / c_perp
        pred = np.sqrt(r)
        dev = abs(ratio - pred) / pred * 100
        out['cases'][str(r)] = {
            'dt': dt, 'c_par': c_par, 'c_perp': c_perp,
            'ratio': ratio, 'sqrt_r': float(pred),
            'deviation_pct': float(dev)}
        print(f'  r={r} (dt={dt}): c_par={c_par:.3f}, c_perp={c_perp:.3f} '
              f'cells/t.u.; ratio={ratio:.3f} vs sqrt(r)={pred:.3f} '
              f'({dev:.2f}% off)', flush=True)
        if r == 4:
            u = snaps[max(snaps)] if snaps else None
            if u is not None:
                ax = axes[0]
                ax.imshow(u.T, origin='lower', cmap='hot', vmin=0, vmax=1)
                ax.contour(u.T, levels=[U_THRESH], colors='cyan',
                           linewidths=1.2)
                ax.set_title(f'r = 4 wavefront (t = {max(snaps):.0f})')
                ax.set_xlabel('x'); ax.set_ylabel('y')
            ax = axes[1]
            times, exs, eys = track
            ax.plot(times, exs, label='x extent (fast)')
            ax.plot(times, eys, label='y extent (slow)')
            ax.set_xlabel('t'); ax.set_ylabel('front extent (cells)')
            ax.set_title('r = 4 front expansion')
            ax.legend()

    ax = axes[2]
    rs = sorted(float(k) for k in out['cases'])
    ratios = [out['cases'][str(int(r))]['ratio'] for r in rs]
    ax.plot(rs, ratios, 'o', color='tab:red', ms=8, label='measured c_par/c_perp')
    rf = np.linspace(1, 8, 100)
    ax.plot(rf, np.sqrt(rf), 'k--', label='sqrt(r)')
    for r, ratio in zip(rs, ratios):
        c = out['cases'][str(int(r))]
        ax.annotate(f"{c['deviation_pct']:.1f}%", (r, ratio),
                    textcoords='offset points', xytext=(8, -12), fontsize=9)
    ax.set_xlabel('anisotropy ratio r = D_par/D_perp')
    ax.set_ylabel('c_par / c_perp')
    ax.set_title('sqrt(r) check')
    ax.legend()

    fig.suptitle('TEST 4a FINAL (de-hacked rd_core): anisotropic target '
                 'wave, Barkley kinetics', fontsize=13)
    fig.tight_layout()
    figpath = 'figures/rd_transfer_aniso_final.png'
    fig.savefig(figpath, dpi=150)
    plt.close(fig)
    print(f'Saved {figpath}')

    jsonpath = 'figures/rd_transfer_aniso_final.json'
    with open(jsonpath, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(f'Saved {jsonpath}')


if __name__ == '__main__':
    main()
