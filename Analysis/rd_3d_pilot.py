"""
3D Oregonator BZ pilot: expanding spherical wave in a uniform medium.

Runs in segments so that mid-plane snapshots can be saved without
modifying the solver core.  Usage:

    ../.venv/bin/python rd_3d_pilot.py [N] [NSTEPS]

Defaults: N=64, NSTEPS=400.
"""

import argparse
import time
import resource
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rd_core_3d import RDSubstrate3D


# Oregonator parameters requested by the user
F = 1.4375
EPS = 0.0501
Q = 0.002
DU = 1.0
DV = 0.0
PHI0 = 0.010
DT = 0.05
DX = 1.0

# Homogeneous rest state for the light-held Oregonator at phi0=0.010.
U_REST = 0.0030821
V_REST = 0.0030821


def build_substrate(N):
    rd = RDSubstrate3D(nx=N, ny=N, nz=N, dx=DX, dt=DT,
                       eps=EPS, q=Q, f=F, Du=DU, Dv=DV, phi=PHI0,
                       clamp_rest=(U_REST, V_REST))
    rd.u[:] = U_REST
    rd.v[:] = V_REST

    # central spherical port
    cx = cy = cz = N // 2
    radius = max(2, N // 20)
    ii, jj, kk = np.mgrid[0:N, 0:N, 0:N]
    port_mask = ((ii - cx)**2 + (jj - cy)**2 + (kk - cz)**2) <= radius**2
    rd.add_port('source', port_mask)

    # probes: centre and a thin spherical shell
    rd.add_probe('centre', port_mask)
    shell_inner, shell_outer = radius + 5, radius + 10
    r = np.sqrt((ii - cx)**2 + (jj - cy)**2 + (kk - cz)**2)
    shell_mask = (r >= shell_inner) & (r <= shell_outer)
    if shell_mask.any():
        rd.add_probe('shell', shell_mask)
    else:
        # fallback for very small grids
        rd.add_probe('shell', port_mask)

    # short central pulse
    rd.fire('source', value=0.8, v_value=0.2, duration=5)
    return rd


def main():
    parser = argparse.ArgumentParser(description='3D Oregonator BZ pilot')
    parser.add_argument('N', nargs='?', type=int, default=64,
                        help='cubic grid size (default 64)')
    parser.add_argument('nsteps', nargs='?', type=int, default=400,
                        help='number of steps to run (default 400)')
    args = parser.parse_args()

    N = args.N
    nsteps = args.nsteps

    print(f'3D Oregonator pilot: grid {N}x{N}x{N}, dt={DT}, dx={DX}')
    print(f'Parameters: f={F}, eps={EPS}, q={Q}, Du={DU}, Dv={DV}, phi0={PHI0}')
    print(f'Rest state: u*={U_REST}, v*={V_REST}')

    diffusion_number = DU * DT / DX**2
    print(f'Diffusion number Du*dt/dx^2 = {diffusion_number:.4f} '
          f'(3D limit 1/6 = {1/6:.4f})')

    rd = build_substrate(N)

    # snapshot schedule: capture mid-plane slices at these completed steps
    snapshot_steps = sorted(set([s for s in [50, 150, 300, nsteps] if 0 < s <= nsteps]))
    snapshots = {}
    probe_history = {name: [] for name in rd.probes}
    time_history = []

    print(f'Running {nsteps} steps...')
    t0 = time.perf_counter()

    prev = 0
    for step_target in snapshot_steps:
        seg = step_target - prev
        data = rd.run(seg)
        prev = step_target
        time_history.extend(data['t'].tolist())
        for name in rd.probes:
            probe_history[name].extend(data[name].tolist())
        zmid = N // 2
        snapshots[step_target] = rd.u[:, :, zmid].copy()
        print(f'  step {rd.t}: max u={rd.u.max():.4f}, shell u={data["shell"][-1]:.4f}')

    elapsed = time.perf_counter() - t0
    max_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    max_rss_mb = max_rss_kb / 1024.0

    print(f'\nRun complete in {elapsed:.1f} s')
    print(f'Peak RSS: {max_rss_mb:.1f} MB')

    t = np.array(time_history)
    print('\nProbe summary:')
    for name in rd.probes:
        series = np.array(probe_history[name])
        print(f'  {name}: max={series.max():.4f}, final={series[-1]:.4f}')

    # save outputs
    figdir = Path('figures')
    figdir.mkdir(exist_ok=True)
    np.savez(figdir / f'rd_3d_pilot_N{N}_series.npz',
             t=t, **{name: np.array(probe_history[name]) for name in rd.probes})

    # snapshot figure
    n_snap = len(snapshots)
    fig, axes = plt.subplots(1, n_snap, figsize=(4 * n_snap, 4))
    if n_snap == 1:
        axes = [axes]
    for ax, (step, slc) in zip(axes, sorted(snapshots.items())):
        im = ax.imshow(slc.T, origin='lower', cmap='inferno',
                       vmin=U_REST, vmax=0.8)
        ax.set_title(f't={step * DT:.1f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f'3D Oregonator mid-plane z={N//2} (N={N})')
    fig.tight_layout()
    fig.savefig(figdir / f'rd_3d_pilot_N{N}_snapshots.png', dpi=150)
    plt.close(fig)

    # time-series figure
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in rd.probes:
        ax.plot(t, np.array(probe_history[name]), label=name)
    ax.axvline(5 * DT, color='gray', linestyle='--', label='pulse end')
    ax.set_xlabel('time')
    ax.set_ylabel('mean u')
    ax.set_title(f'3D Oregonator probes (N={N})')
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / f'rd_3d_pilot_N{N}_probes.png', dpi=150)
    plt.close(fig)

    print(f'Figures saved to {figdir}/')

    # text report
    report_path = figdir / f'rd_3d_pilot_N{N}_report.txt'
    with open(report_path, 'w') as f:
        f.write('3D Oregonator pilot report\n')
        f.write(f'grid: {N}x{N}x{N}\n')
        f.write(f'dt: {DT}\n')
        f.write(f'steps: {nsteps}\n')
        f.write(f'physical time: {nsteps * DT:.2f}\n')
        f.write(f'runtime: {elapsed:.1f} s\n')
        f.write(f'peak RSS: {max_rss_mb:.1f} MB\n')
        f.write(f'diffusion number: {diffusion_number:.4f} '
                f'(limit 1/6 = {1/6:.4f})\n')
        for name in rd.probes:
            series = np.array(probe_history[name])
            f.write(f'probe {name}: max={series.max():.4f}, '
                    f'final={series[-1]:.4f}\n')
    print(f'Report saved to {report_path}')


if __name__ == '__main__':
    main()
