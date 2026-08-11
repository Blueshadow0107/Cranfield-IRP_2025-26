"""
rd_darkspot_driver_multi.py

Multi-spot, multi-flash driver for dark-spot input experiments.
Each spot is a local temporary reduction of phi; flashes can be scheduled
independently for each spot, so pulse trains and multi-input gates are
supported without clamped ports.
"""

import numpy as np


def build_phi_schedule(rd, phi0, spots, nsteps):
    """
    Build an (nsteps, nx, ny) phi schedule from spot flash definitions.

    Parameters
    ----------
    rd : RDSubstrate
    phi0 : float
        Background phi value.
    spots : list of dict
        Each dict has keys:
            'mask'      : (nx, ny) bool array, cells that receive the flash
            'times'     : iterable of flash start steps
            'duration'  : int, flash length in steps
            'phi_dark'  : float, phi inside the spot during the flash
    nsteps : int
        Total number of solver steps.

    Returns
    -------
    phi : (nsteps, nx, ny) float array
    """
    phi = np.full((nsteps, rd.nx, rd.ny), phi0, dtype=float)
    for spot in spots:
        mask = spot['mask']
        duration = int(spot['duration'])
        phi_dark = float(spot['phi_dark'])
        for t0 in spot['times']:
            t0 = int(t0)
            phi[t0:t0 + duration, mask] = phi_dark
    return phi


def run_darkspot_multi(rd, phi0, spots, nsteps, probes=None):
    """
    Run `rd` for `nsteps` with a scheduled multi-spot dark-flash input.

    Parameters
    ----------
    rd : RDSubstrate
        Pre-configured substrate (walls, diffusion tensor, initial state).
    phi0 : float
        Background phi value.
    spots : list of dict
        Spot flash definitions, see `build_phi_schedule`.
    nsteps : int
        Total number of steps to run.
    probes : dict[str -> bool array], optional
        If given, record mean u at each probe every step.

    Returns
    -------
    dict with 't' and one array per probe name.
    """
    if probes is None:
        probes = rd.probes

    schedule = build_phi_schedule(rd, phi0, spots, nsteps)
    series = {name: [] for name in probes}
    ts = []

    for step in range(nsteps):
        rd.set_phi(schedule[step])
        rd._step()
        rd.t += 1
        ts.append(rd.t * rd.dt)
        for name, mask in probes.items():
            series[name].append(float(rd.u[mask].mean()))

    out = {'t': np.array(ts, dtype=float)}
    for name in probes:
        out[name] = np.array(series[name], dtype=float)
    return out
