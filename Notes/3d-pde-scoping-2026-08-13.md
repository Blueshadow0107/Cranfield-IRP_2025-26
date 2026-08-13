# 3D Oregonator BZ scoping — 2026-08-13

## Question
What is needed to extend the current 2D Oregonator work to 3D, and what does it cost?

## What the graph simulator needs

Almost nothing. `rd_graph_sim.py` is dimension-agnostic:
- Nodes = pores / junctions.
- Edges = throats with transit time `L / c`.
- Rules (refractory, inhibition, routing) depend only on arrival times, not on spatial dimension.

To move to 3D we would only need to re-measure the local transfer parameters (`c`, `tau`, inhibition window, amplitude) in 3D PDE runs. The graph structure and event loop stay the same.

## What the PDE solver needs

`rd_core.py` is hard-coded 2D (`u.shape == (nx, ny)`). A 3D version needs:

1. **Array shape:** `(nx, ny, nz)` instead of `(nx, ny)`.
2. **Diffusion stencil:** 7-point Laplacian in divergence form on 6 faces per cell (x, y, z).
3. **Stability limit:** explicit diffusion in 3D requires
   `dt <= dx^2 / (6 * D_max)`
   instead of the 2D limit `dt <= dx^2 / (4 * D_max)`. For `D = 1`, `dx = 1`, the safe `dt` drops from `0.25` to `~0.167` (or keep `dt = 0.05` as now, which is very safe).
4. **Wall mask:** 3D boolean mask; no-flux faces touch any wall cell.
5. **Ports / probes:** 3D boolean masks.
6. **Phi field:** 3D scalar field. This is the experimental blocker (see below).
7. **Memory:** a 256^3 domain stores ~16.7 M cells. `u` and `v` together need ~270 MB in float64; with work arrays and tensor components, expect ~1–2 GB. Still feasible on a workstation.
8. **Compute:** 3D has ~1.5× more faces per cell than 2D and ~N^(3/2) more surface for the same linear size. A 256^3 run is roughly an order of magnitude slower per step than 256^2. A short 3D transfer test (a few hundred steps) would take minutes to tens of minutes.

## The real blocker: light control in the bulk

In the current 2D work, `phi(x, y)` is a projected light mask. We know the suppression field everywhere because the domain is a thin slab.

In a 3D bulk reactor:
- A projected 2D mask only controls the illuminated surface (or thin surface layers).
- Light is absorbed/scattered by the BZ medium and the porous scaffold, so `phi(x, y, z)` in the interior is unknown.
- Without a known `phi` field, the local kinetics are uncontrolled and the 2D-calibrated operators do not apply.

Possible ways around this:
1. **Thin-slab geometry:** keep the third dimension small so the light field is approximately uniform. This is essentially a thick 2D problem.
2. **Multiple projections / tomographic illumination:** illuminate from several directions and solve the inverse problem to create a volumetric pattern. Non-trivial experimentally.
3. **Use chemical gradients instead of light:** a 3D chemical inhibitor distribution. Harder to pattern dynamically.
4. **Two-photon or confocal excitation:** expensive and slow for macroscopic volumes.

## Practical recommendation

Stay in 2D for local operator calibration and graph-simulator validation, because it is the only geometry where the light field is fully known. Frame 3D as a direct extension once (i) a volumetric illumination strategy is chosen and (ii) the graph simulator has been shown to compose 2D operators correctly.

The 2D results are therefore not a toy model — they are the controlled micro-scale calibration layer that any 3D porous-medium prediction must rest on.

## Cost estimate for a single 3D channel test

If we do proceed with a thin-slab 3D test:
- Grid: 64 × 64 × 8 (thin in z) ≈ 33k cells, comparable to a 180^2 2D run.
- Steps: 400–800.
- Estimated wall time: a few minutes on the current machine.
- A full 256^3 run: ~1–2 hours for a short test, several hours for a network simulation.

## Next decision

Before writing `rd_core_3d.py`, confirm which experimental light-delivery strategy we are assuming in 3D. The code change is straightforward; the physical boundary condition is not.
