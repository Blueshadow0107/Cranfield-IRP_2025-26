"""rd_demux_baseline_verify.py -- verify the demux baseline animation export.

Reads the .pvd collections produced by rd_demux_baseline_anim_fixed.py using
pyvista, checks that every frame loads, reports the field range, and optionally
writes a screenshot of the middle frame of each pattern.

Usage:
    cd Analysis
    ../.venv/bin/python rd_demux_baseline_verify.py
"""
import os
from pathlib import Path

import numpy as np
import pyvista as pv

FIG = Path(__file__).parent / 'figures' / 'demux_baseline_anim'
PATTERNS = ['01', '10', '11']


def check_pattern(tag):
    pvd = FIG / f'pattern_{tag}.pvd'
    if not pvd.exists():
        print(f'[missing] {pvd}')
        return False

    reader = pv.get_reader(str(pvd))
    n = reader.number_time_points
    print(f'pattern ({tag}): {n} time points, t in [{reader.time_values[0]}, {reader.time_values[-1]}]')

    def get_u(dataset):
        if isinstance(dataset, pv.MultiBlock):
            block = dataset[0]
        else:
            block = dataset
        return block['u']

    u_min_all, u_max_all = np.inf, -np.inf
    for i in range(n):
        reader.set_active_time_value(reader.time_values[i])
        grid = reader.read()
        u = get_u(grid)
        u_min_all = min(u_min_all, float(u.min()))
        u_max_all = max(u_max_all, float(u.max()))

    print(f'  u global range: [{u_min_all:.4f}, {u_max_all:.4f}]')

    # Screenshot of the middle frame
    mid = n // 2
    reader.set_active_time_value(reader.time_values[mid])
    grid = reader.read()
    if isinstance(grid, pv.MultiBlock):
        grid = grid[0]
    plotter = pv.Plotter(off_screen=True, window_size=(600, 600))
    plotter.add_mesh(grid, scalars='u', cmap='viridis',
                     clim=[0.0, 0.8], show_scalar_bar=True)
    plotter.view_xy()
    img_path = FIG / f'pattern_{tag}_preview.png'
    plotter.screenshot(str(img_path))
    plotter.close()
    print(f'  preview saved: {img_path}')
    return True


def main():
    ok = True
    for tag in PATTERNS:
        ok &= check_pattern(tag)
    if ok:
        print('[ok] all baseline animations are ParaView-ready')
    else:
        print('[fail] some patterns are missing or unreadable')


if __name__ == '__main__':
    main()
