"""
Joint space-time convergence map: free-medium pulse speed on a 3x3 grid
of (DX, DT) combinations, for both kinetics.

Design: DX in {1.0, 0.5, 0.25}; per DX, DT = 0.05*DX^2 * {1, 0.5, 0.25},
i.e. diffusion numbers {0.05, 0.025, 0.0125} -- all stable for explicit
diffusion (limit 0.25).  Complements the two separate studies in
rd_verification.py (grid at fixed diffusion number 0.05; timestep at
fixed DX=1) by showing the error surface: if speed depends mostly on DX
the discretisation error is space-dominated; if it moves with DT at
fixed DX there is a temporal component too.

Reuses rd_verification.speed_run (physical domain 200x40, physical port
and probe positions, diffusion-number held per combination).

Outputs: Analysis/figures/rd_convergence_map.json (+ stdout table)
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rd_verification import KINS, speed_run   # noqa: E402

DXS = [1.0, 0.5, 0.25]
DT_FACTORS = [1.0, 0.5, 0.25]       # multiplier on 0.05*DX^2
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'figures', 'rd_convergence_map.json')


def main():
    results = {}
    for kin in KINS:
        results[kin] = []
        for DX in DXS:
            for fac in DT_FACTORS:
                DT = 0.05 * DX * DX * fac
                r = speed_run(kin, DX, DT)
                r['dt_factor'] = fac
                results[kin].append(r)
                print(f"{kin:10s} DX={DX:<5} DT={DT:.7f} dn={r['diffusion_number']:.4f} "
                      f"c={r['c_phys']:.4f} len/t.u. "
                      f"({r['grid'][0]}x{r['grid'][1]}, {r['nsteps']} steps)",
                      flush=True)
    with open(OUT, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f'saved {OUT}')

    # compact summary table
    print('\nc_phys (length units / t.u.):')
    for kin in KINS:
        print(f'  {kin}:')
        print('         dn=0.05     dn=0.025    dn=0.0125')
        for i, DX in enumerate(DXS):
            row = [r for r in results[kin] if r['DX'] == DX]
            row.sort(key=lambda r: -r['DT'])
            print(f'  DX={DX:<5}', '  '.join(f"{r['c_phys']:.4f}" for r in row))


if __name__ == '__main__':
    main()
