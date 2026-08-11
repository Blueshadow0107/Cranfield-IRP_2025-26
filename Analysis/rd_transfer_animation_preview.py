"""Generate single-frame preview PNGs for the transfer-function animations."""
from pathlib import Path

import pyvista as pv

FIG = Path(__file__).parent / 'figures' / 'transfer_animations'
CASES = ['channel_w16', 'frequency_train', 'logic_A_only', 'logic_A_and_B',
         'anisotropic_r4']


def preview(name, timestep_index=None):
    pvd = FIG / f'{name}.pvd'
    if not pvd.exists():
        print(f'[missing] {pvd}')
        return
    reader = pv.get_reader(str(pvd))
    if timestep_index is None:
        timestep_index = len(reader.time_values) // 3
    reader.set_active_time_value(reader.time_values[timestep_index])
    grid = reader.read()
    if isinstance(grid, pv.MultiBlock):
        grid = grid[0]

    plotter = pv.Plotter(off_screen=True, window_size=(640, 480))
    plotter.add_mesh(grid, scalars='u', cmap='viridis', clim=[0.0, 0.8],
                     show_scalar_bar=True, scalar_bar_args={'title': 'u'})
    plotter.view_xy()
    out = FIG / f'{name}_preview.png'
    plotter.screenshot(str(out))
    plotter.close()
    print(f'[saved] {out}')


def main():
    for name in CASES:
        preview(name)


if __name__ == '__main__':
    main()
