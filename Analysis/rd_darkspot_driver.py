"""
rd_darkspot_driver.py

Small helper for running simulations with a one-shot dark-spot input drive.
The spot is a local temporary reduction of the light-suppression field phi;
while it is dark the patch acts as a pacemaker and emits a pulse into the
surrounding excitable medium.

This avoids using clamped ports and is the experimentally honest input for
photosensitive BZ experiments.
"""

import numpy as np


def run_darkspot(rd, spot_mask, phi0, phi_dark, duration_steps, nsteps,
                 probes=None):
    """
    Run `rd` for `nsteps` with a dark-spot flash.

    Parameters
    ----------
    rd : RDSubstrate
        Pre-configured substrate (walls, diffusion tensor, initial state).
    spot_mask : (nx, ny) bool array
        Cells that receive the dark flash.
    phi0 : float
        Background phi value.
    phi_dark : float
        Phi value inside the spot during the flash.
    duration_steps : int
        Length of the flash in solver steps.
    nsteps : int
        Total number of steps to run.
    probes : dict[str -> bool array], optional
        If given, record mean u at each probe every step.  If omitted,
        record whatever probes are already registered on `rd`.

    Returns
    -------
    dict with 't' and one array per probe name.
    """
    if probes is None:
        probes = rd.probes

    series = {name: [] for name in probes}
    ts = []
    base_phi = np.full((rd.nx, rd.ny), phi0, dtype=float)

    for step in range(nsteps):
        phi = base_phi.copy()
        if step < duration_steps:
            phi[spot_mask] = phi_dark
        rd.set_phi(phi)
        rd._step()
        rd.t += 1
        ts.append(rd.t * rd.dt)
        for name, mask in probes.items():
            series[name].append(float(rd.u[mask].mean()))

    out = {'t': np.array(ts, dtype=float)}
    for name in probes:
        out[name] = np.array(series[name], dtype=float)
    return out
