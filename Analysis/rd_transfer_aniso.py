"""
TEST 4 -- Anisotropic tensor diffusion (no walls in 4a/4b).

KINETICS: run primarily on the excitable Barkley model (a=0.75, b=0.01,
eps=0.02, Dv=0; see TEST 2 header for why the Oregonator baseline is
unsuitable for clean pulse experiments).  4a is additionally cross-checked
on the Oregonator baseline (leading-edge speeds only).

4a  Uniform anisotropic medium, D_perp = 1, D_par = r (fast axis = x),
    r in {1, 2, 4} at dt = 0.05 plus r = 8 at dt = 0.02.  A central disk
    pulse is seeded; the wavefront extent along x and y is tracked over
    time giving axis speeds c_par, c_perp.  Check c_par/c_perp ~ sqrt(r).

4b  r = 4, plane waves propagating in +x with the TENSOR tilted by theta
    relative to the propagation direction (theta in {0, 30, 45, 60, 90}).
    Equivalent to a wave travelling at angle theta to the fast axis, but
    keeps the front planar and the measurement 1-D.  Effective diffusivity
    along the propagation direction n is D_eff = n^T D n, so theory gives
        c(theta) = sqrt(c_par^2 cos^2(theta) + c_perp^2 sin^2(theta)).

4c  Channel (width 24, walls) through a central rectangular anisotropic
    patch (x in [96, 160], full channel height; r = 4, D_perp = 1, fast
    axis at angle theta to the channel axis; isotropic D = 1 outside).
    Transmission delay across the patch vs theta.

Outputs: Analysis/figures/rd_transfer_aniso_{a,b,c}.png and
         Analysis/figures/rd_transfer_aniso.json.
"""

import json
import numpy as np
import matplotlib.pyplot as plt

from rd_core import RDSubstrate

NX, NY = 256, 256
CX, CY = NX // 2, NY // 2
U_THRESH = 0.5
KIN = dict(kinetics='barkley', eps=0.02, Dv=0.0)
KIN_OREG = dict(kinetics='oregonator')   # baseline cross-check (4a only)


def disk_mask(cx, cy, r):
    X, Y = np.mgrid[0:NX, 0:NY]
    return (X - cx)**2 + (Y - cy)**2 <= r**2


def tensor_components(r, theta_deg):
    """Tensor with eigenvalues (r, 1), fast axis at theta to the x-axis."""
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    Dxx = r * c**2 + 1.0 * s**2
    Dyy = r * s**2 + 1.0 * c**2
    Dxy = (r - 1.0) * s * c
    return Dxx, Dyy, Dxy


def front_extents(u, thresh=U_THRESH):
    """Outermost front distance from centre along +x, -x, +y, -y."""
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


# ---------------------------------------------------------------------------
def run_4a(r, dt, nsteps, snap_times=(), kin=None):
    rd = RDSubstrate(nx=NX, ny=NY, dt=dt, **(kin or KIN))
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


def part_a():
    print('--- 4a: axis speeds vs anisotropy ratio (barkley) ---')
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for r, dt, nsteps in [(1, 0.05, 900), (2, 0.05, 700),
                          (4, 0.05, 500), (8, 0.02, 900)]:
        c_par, c_perp, snaps, track = run_4a(r, dt, nsteps,
                                             snap_times=(20, 40, 60))
        ratio = c_par / c_perp
        pred = np.sqrt(r)
        dev = abs(ratio - pred) / pred * 100
        out[str(r)] = {'c_par': c_par, 'c_perp': c_perp,
                       'ratio': ratio, 'sqrt_r': float(pred),
                       'deviation_pct': float(dev)}
        print(f'  r={r}: c_par={c_par:.3f}, c_perp={c_perp:.3f} '
              f'cells/t.u.; ratio={ratio:.3f} vs sqrt(r)={pred:.3f} '
              f'({dev:.1f}% off)')
        if r == 4:
            u = snaps[max(snaps)] if snaps else None
            if u is not None:
                ax = axes[0]
                ax.imshow(u.T, origin='lower', cmap='hot', vmin=0, vmax=1)
                cs = ax.contour(u.T, levels=[U_THRESH], colors='cyan',
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
    fig.suptitle('TEST 4a: anisotropic target wave', fontsize=13)
    fig.tight_layout()
    fig.savefig('figures/rd_transfer_aniso_a.png', dpi=150)
    print('Saved figures/rd_transfer_aniso_a.png')

    # Oregonator-baseline cross-check.  The baseline medium is oscillatory
    # (see TEST 2 header), so extent tracking is polluted by the trailing
    # self-fired train; use FIRST-PASSAGE times at fixed distances along
    # each axis instead (leading front only).
    print('--- 4a cross-check: oregonator baseline (first-passage) ---')
    oreg = {}
    for r, dt, nsteps in [(1, 0.05, 700), (4, 0.05, 400)]:
        rd = RDSubstrate(nx=NX, ny=NY, dt=dt, **KIN_OREG)
        Dxx, Dyy, Dxy = tensor_components(r, 0.0)
        rd.set_diffusion_tensor(Dxx, Dyy, Dxy)
        seed = disk_mask(CX, CY, 25)
        rd.u[seed] = 0.8
        rd.v[seed] = 0.2
        rd.add_probe('x40', disk_mask(CX + 40, CY, 2))
        rd.add_probe('x80', disk_mask(CX + 80, CY, 2))
        rd.add_probe('y40', disk_mask(CX, CY + 40, 2))
        rd.add_probe('y80', disk_mask(CX, CY + 80, 2))
        data = rd.run(nsteps)

        def cross(s):
            idx = np.nonzero(s >= U_THRESH)[0]
            return float(idx[0] * dt) if len(idx) else None
        speeds = {}
        for ax in ('x', 'y'):
            t1, t2 = cross(data[ax + '40']), cross(data[ax + '80'])
            speeds[ax] = 40.0 / (t2 - t1) if (t1 and t2) else None
        c_par, c_perp = speeds['x'], speeds['y']
        ratio = c_par / c_perp
        pred = np.sqrt(r)
        oreg[str(r)] = {'c_par': c_par, 'c_perp': c_perp, 'ratio': ratio,
                        'sqrt_r': float(pred),
                        'deviation_pct': float(abs(ratio - pred) / pred * 100)}
        print(f'  r={r}: c_par={c_par:.3f}, c_perp={c_perp:.3f}, '
              f'ratio={ratio:.3f} vs sqrt(r)={pred:.3f}')
    out['oregonator_crosscheck'] = oreg
    return out


# ---------------------------------------------------------------------------
def run_4b(theta, r=4, dt=0.05, nsteps=1400):
    rd = RDSubstrate(nx=NX, ny=NY, dt=dt, **KIN)
    Dxx, Dyy, Dxy = tensor_components(r, theta)
    rd.set_diffusion_tensor(Dxx, Dyy, Dxy)
    slab = np.zeros((NX, NY), bool)
    slab[36:44, :] = True
    rd.u[slab] = 0.8
    rd.v[slab] = 0.2
    times, fronts = [], []
    chunk = 10
    done = 0
    while done < nsteps:
        rd.run(min(chunk, nsteps - done))
        done += chunk
        col_max = (rd.u > U_THRESH).any(axis=1)
        xs = np.nonzero(col_max)[0]
        fronts.append(xs.max() if len(xs) else 0)
        times.append(rd.t * dt)
    # speed from front position vs time (window 60..170 cells)
    sel = [(t, f) for t, f in zip(times, fronts) if 60 <= f <= 170]
    t = np.array([s[0] for s in sel]); f = np.array([s[1] for s in sel])
    slope, _ = np.polyfit(t, f, 1)
    assert not np.isnan(rd.u).any(), 'NaN blowup'
    return float(slope)


def part_b(c_par, c_perp):
    print('--- 4b: speed vs propagation angle (r=4) ---')
    thetas = [0, 30, 45, 60, 90]
    rows = []
    for th in thetas:
        c = run_4b(th)
        cth = np.sqrt(c_par**2 * np.cos(np.deg2rad(th))**2
                      + c_perp**2 * np.sin(np.deg2rad(th))**2)
        dev = abs(c - cth) / cth * 100
        rows.append({'theta': th, 'c_meas': c, 'c_theory': float(cth),
                     'deviation_pct': float(dev)})
        print(f'  theta={th:3d}: c={c:.3f} vs theory {cth:.3f} '
              f'({dev:.1f}% off)')
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    th_fine = np.linspace(0, 90, 100)
    ax.plot(th_fine, np.sqrt(c_par**2 * np.cos(np.deg2rad(th_fine))**2
                             + c_perp**2 * np.sin(np.deg2rad(th_fine))**2),
            'k--', label='theory sqrt(c_par^2 cos^2 + c_perp^2 sin^2)')
    ax.plot([r['theta'] for r in rows], [r['c_meas'] for r in rows],
            'o', color='tab:red', label='measured')
    ax.set_xlabel('angle to fast axis (deg)')
    ax.set_ylabel('wave speed (cells / t.u.)')
    ax.set_title('TEST 4b: c(theta) in anisotropic medium (r=4)')
    ax.legend()
    fig.tight_layout()
    fig.savefig('figures/rd_transfer_aniso_b.png', dpi=150)
    print('Saved figures/rd_transfer_aniso_b.png')
    return rows


# ---------------------------------------------------------------------------
def run_4c(theta, r=4, width=24, nsteps=2500):
    rd = RDSubstrate(nx=NX, ny=NY, **KIN)
    w2 = width // 2
    wall = np.ones((NX, NY), bool)
    wall[:, CY - w2:CY + w2] = False
    rd.set_walls(wall)

    Dxx = np.ones((NX, NY)); Dyy = np.ones((NX, NY))
    Dxy = np.zeros((NX, NY))
    if theta is not None:
        dxx, dyy, dxy = tensor_components(r, theta)
        patch = np.zeros((NX, NY), bool)
        patch[96:160, CY - w2:CY + w2] = True
        Dxx[patch] = dxx; Dyy[patch] = dyy; Dxy[patch] = dxy
    rd.set_diffusion_tensor(Dxx, Dyy, Dxy)

    port = np.zeros((NX, NY), bool)
    port[0:18, CY - w2:CY + w2] = True
    rd.add_port('in', port)
    rd.add_probe('pre', disk_mask(56, CY, 2) & ~wall)
    rd.add_probe('post', disk_mask(200, CY, 2) & ~wall)
    rd.fire('in')
    data = rd.run(nsteps)

    def cross(s):
        idx = np.nonzero(s >= U_THRESH)[0]
        return int(idx[0]) if len(idx) else None
    t_pre, t_post = cross(data['pre']), cross(data['post'])
    delay = (t_post - t_pre) if (t_pre is not None and t_post is not None) \
        else None
    return delay, float(data['post'].max())


def part_c():
    print('--- 4c: channel through anisotropic patch (r=4) ---')
    base_delay, _ = run_4c(None)
    print(f'  isotropic baseline delay = {base_delay} steps')
    rows = []
    for th in [0, 30, 45, 60, 90]:
        delay, peak = run_4c(th)
        blocked = delay is None
        rows.append({'theta': th, 'delay_steps': delay,
                     'extra_delay_vs_baseline':
                         (delay - base_delay) if not blocked else None,
                     'blocked': blocked, 'peak_post': peak})
        msg = f'{delay} steps ({delay - base_delay:+d} vs baseline)' \
              if not blocked else f'BLOCKED (peak_post={peak:.3f})'
        print(f'  theta={th:3d}: delay {msg}')
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ok = [r for r in rows if not r['blocked']]
    ax.plot([r['theta'] for r in ok], [r['delay_steps'] for r in ok], 'o-')
    ax.axhline(base_delay, color='k', ls='--', lw=0.8,
               label=f'isotropic baseline ({base_delay})')
    ax.set_xlabel('patch fast-axis angle to channel (deg)')
    ax.set_ylabel('pre->post probe delay (steps)')
    ax.set_title('TEST 4c: anisotropic patch in a channel (r=4)')
    ax.legend()
    fig.tight_layout()
    fig.savefig('figures/rd_transfer_aniso_c.png', dpi=150)
    print('Saved figures/rd_transfer_aniso_c.png')
    return {'baseline_delay': base_delay, 'sweep': rows}


# ---------------------------------------------------------------------------
def main():
    results = {}
    a = part_a()
    results['4a'] = a
    # use measured r=4 axis speeds as c_par / c_perp reference for 4b theory
    c_par, c_perp = a['4']['c_par'], a['4']['c_perp']
    results['4b'] = part_b(c_par, c_perp)
    results['4c'] = part_c()
    with open('figures/rd_transfer_aniso.json', 'w') as fh:
        json.dump(results, fh, indent=2)
    print('Saved figures/rd_transfer_aniso.json')


if __name__ == '__main__':
    main()
