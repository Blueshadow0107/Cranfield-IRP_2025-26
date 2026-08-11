"""
Fast phi-space characterisation for the light-held Oregonator.

Physics: f=1.4375, eps=0.05014844822490394, q=0.002.
Point-kinetics classification is cheap and done for a fine phi grid.
Spatial propagation is expensive, so it is checked only at sparse
boundary points (and reused from existing rd_phi_barrier.json where
available).

The figure shows:
  - rest state u* vs phi
  - max Re(eigenvalue) vs phi (oscillatory when >0)
  - point-kinetics regime
  - spatial propagation flag

Three relevant regimes:
  oscillatory (phi <~ 0.005)
  excitable-propagating (0.006 <= phi <~ 0.028)
  subexcitable/non-propagating (phi >~ 0.030)

Output: figures/rd_phi_regime_map.json and .png
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
from rd_core import RDSubstrate

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG, exist_ok=True)

F = 1.4375
EPS = 0.05014844822490394
Q = 0.002
DT = 0.05
U_THRESH = 0.5

PHI_GRID = np.round(np.concatenate([
    np.linspace(0.0, 0.010, 41),
    np.linspace(0.012, 0.030, 19),
    np.linspace(0.032, 0.050, 10)
]), 6)
PHI_GRID = np.unique(PHI_GRID)


def rest_state_and_eigs(phi):
    def Fn(u):
        return u - u**2 - (F * u + phi) * (u - Q) / (u + Q)
    ug = np.concatenate(([0.0], np.logspace(-7, np.log10(1.2), 4000)))
    vals = Fn(ug)
    ustar = None
    for i in range(len(ug) - 1):
        if vals[i] > 0.0 and vals[i + 1] <= 0.0:
            ustar = brentq(Fn, ug[i], ug[i + 1], xtol=1e-14, rtol=1e-12)
            break
    if ustar is None:
        return None, None
    g = (ustar - Q) / (ustar + Q)
    dg = 2.0 * Q / (ustar + Q)**2
    s = F * ustar + phi
    Fu = (1.0 - 2.0 * ustar - s * dg) / EPS
    Fv = -(F * g) / EPS
    tr = Fu - 1.0
    det = -Fu - Fv
    disc = tr * tr - 4.0 * det
    if disc >= 0.0:
        r = np.sqrt(disc)
        eigs = np.array([0.5 * (tr + r), 0.5 * (tr - r)])
    else:
        r = np.sqrt(-disc)
        eigs = np.array([0.5 * tr + 0.5j * r, 0.5 * tr - 0.5j * r])
    return ustar, float(eigs.real.max())


def point_regime(phi, ustar):
    def rhs(t, y):
        u, v = y
        return [(u - u*u - (F * v + phi) * (u - Q) / (u + Q)) / EPS, u - v]
    sol = solve_ivp(rhs, (0.0, 500.0), [0.2, ustar], method='LSODA',
                    rtol=1e-7, atol=1e-10, t_eval=np.linspace(0, 500, 2501))
    u = sol.y[0]
    fired = bool(u.max() > 0.5)
    tail = u[int(0.8 * len(u)):]
    returned = bool(u[-1] < 0.05 and tail.max() < 0.3)
    if fired and returned:
        return 'excitable'
    if not fired:
        return 'subexcitable'
    return 'oscillatory'


def spatial_propagation(phi, ustar, nx=120, ny=32, width=12):
    rd = RDSubstrate(nx=nx, ny=ny, dt=DT, kinetics='oregonator',
                     f=F, eps=EPS, clamp_rest=(ustar, ustar))
    rd.set_phi(np.full((nx, ny), phi))
    wall = np.ones((nx, ny), bool)
    cy = ny // 2
    wall[:, cy - width // 2:cy + width // 2] = False
    rd.set_walls(wall)
    rd.u[~wall] = ustar
    rd.v[~wall] = ustar
    port = np.zeros((nx, ny), bool)
    port[0:18, cy - width // 2:cy + width // 2] = True
    rd.add_port('in', port)
    X, Y = np.mgrid[0:nx, 0:ny]
    def probe(x0):
        return ((X - x0)**2 + (Y - cy)**2 <= 4) & ~wall
    rd.add_probe('p1', probe(40))
    rd.add_probe('p2', probe(80))
    rd.fire('in', value=0.8, v_value=0.2, duration=30)
    data = rd.run(500)
    s2 = data['p2']
    above = s2 >= U_THRESH
    crossed = bool(above.any())
    peak = float(s2.max())
    first = float(np.argmax(above) * DT) if crossed else None
    return {'crossed_40_cells': crossed, 'peak_p2': peak, 'first_arrival_tu': first}


def main():
    # Load existing spatial propagation bracket if available
    existing = {}
    barrier_path = os.path.join(FIG, 'rd_phi_barrier.json')
    if os.path.exists(barrier_path):
        with open(barrier_path) as fh:
            d = json.load(fh)
        for k, v in d.items():
            existing[float(k)] = v['transmitted']
        print(f"[info] loaded existing propagation data from {barrier_path}")

    results = []
    spatial_check_phis = [0.004, 0.006, 0.010, 0.020, 0.025, 0.028, 0.030, 0.032, 0.040, 0.050]
    for phi in PHI_GRID:
        ustar, max_re = rest_state_and_eigs(phi)
        if ustar is None:
            results.append({'phi': float(phi), 'regime': 'no_rest'})
            continue
        preg = point_regime(phi, ustar)
        if phi in spatial_check_phis or phi in existing:
            if phi in existing:
                prop = {'crossed_40_cells': existing[phi], 'peak_p2': None, 'first_arrival_tu': None}
            else:
                prop = spatial_propagation(phi, ustar)
        else:
            prop = None
        rec = {
            'phi': float(phi),
            'u_star': float(ustar),
            'max_re_eig': float(max_re),
            'point_regime': preg,
        }
        if prop is not None:
            rec.update({
                'propagation_crossed_40_cells': prop['crossed_40_cells'],
                'propagation_peak_p2': prop['peak_p2'],
                'propagation_first_arrival_tu': prop['first_arrival_tu']
            })
        results.append(rec)
        extra = f" propagates={prop['crossed_40_cells']}" if prop else ""
        print(f"phi={phi:.4f} u*={ustar:.5f} maxRe={max_re:+.3f} "
              f"point={preg:14s}{extra}", flush=True)

    with open(os.path.join(FIG, 'rd_phi_regime_map.json'), 'w') as fh:
        json.dump({'config': {'f': F, 'eps': EPS, 'q': Q, 'dt': DT}, 'data': results}, fh, indent=2)

    # Plot
    phis = np.array([r['phi'] for r in results if 'u_star' in r])
    rests = np.array([r['u_star'] for r in results if 'u_star' in r])
    maxres = np.array([r['max_re_eig'] for r in results if 'u_star' in r])
    # spatial flags at sparse points
    prop_data = [(r['phi'], r.get('propagation_crossed_40_cells', None),
                  r.get('propagation_peak_p2', 0.0)) for r in results if 'u_star' in r]
    prop_phis = np.array([p[0] for p in prop_data if p[1] is not None])
    prop_peaks = np.array([p[2] for p in prop_data if p[1] is not None])
    prop_flags = np.array([p[1] for p in prop_data if p[1] is not None])

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    ax = axes[0]
    ax.plot(phis, rests, 'k.-')
    ax.set_ylabel('rest state $u^*$')
    ax.set_title('Light-held Oregonator parameter map ($f=1.4375$, $\\varepsilon=0.0501$)')

    ax = axes[1]
    ax.plot(phis, maxres, 'b.-')
    ax.axhline(0, color='r', ls='--', lw=1)
    ax.set_ylabel('max Re(eig)')
    ax.set_yscale('symlog', linthresh=1.0)

    ax = axes[2]
    ax.plot(phis, np.zeros_like(phis), 'k-', lw=0.5)
    # scatter: propagate (1) vs not (0), peak as marker size/color
    colors = ['green' if f else 'red' for f in prop_flags]
    ax.scatter(prop_phis, prop_flags.astype(float) * 0.5 + 0.25, c=colors, s=50, zorder=3)
    if len(prop_peaks):
        ax2 = ax.twinx()
        ax2.plot(prop_phis, prop_peaks, 'c.--', alpha=0.7, label='probe peak')
        ax2.set_ylabel('probe peak $u$')
        ax2.set_ylim(0, 0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel('propagates')
    ax.set_xlabel('background illumination $\\phi$')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['no', 'yes'])

    # shade regimes
    for ax in axes:
        ax.axvspan(0, 0.005, alpha=0.08, color='red')
        ax.axvspan(0.005, 0.028, alpha=0.08, color='green')
        ax.axvspan(0.028, 0.050, alpha=0.08, color='blue')

    for ax in axes:
        ax.set_xlim(PHI_GRID.min(), PHI_GRID.max())
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'rd_phi_regime_map.png'), dpi=150)
    print(f"[done] saved {FIG}/rd_phi_regime_map.*")


if __name__ == '__main__':
    main()
