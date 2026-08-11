# Discussion points for supervisor meetings

Running list of open questions to raise with Prof. Guo / Prof. Tsoutsanis.
Newest first. Add items as they come up; strike through when resolved.

## 1. Pre-asymptotic grid convergence of the excitable front (2026-07-30)

**The observation.** Free-medium pulse speed (Barkley kinetics) at DX = 1.0, 0.5, 0.25
gives 4.1507 / 4.4391 / 4.6031 cells/t.u. The observed convergence order is only
~0.81 (expected 2 for our central-difference scheme): the excitable front is
2–3 cells thick at DX=1 and still marginally resolved at DX=0.25, so we are
"pre-asymptotic" — refinement is still fixing the front shape rather than
polishing a resolved solution. Richardson extrapolation gives c_inf ≈ 4.82,
i.e. a ~14% speed deficit at our working DX=1, but with a caveat on the
extrapolation itself. Oregonator is much better behaved (~3%, thicker front).
Timestep error is negligible (<=1.5%); the joint (DX,DT) map confirms the error
is space-dominated and separable.

**The question for them.** What is the accepted way to handle this at MSc level?

- (a) Accept + report the error bar (all our measurements share the working
  resolution, so comparative results are internally consistent).
- (b) One DX=0.5 production run on HPC for the headline numbers (~8x cost).
- (c) Higher-order spatial discretisation (4th-order fluxes) — helps the error
  constant but cannot fully fix a 2–3-cell front.
- (d) Cross-check coarse-grid speeds against eikonal theory (Keener–Tyson),
  i.e. predict the continuum speed analytically instead of refining toward it.

My current lean: (a) for all comparative results, (b) for the 2–3 headline
numbers once HPC access is confirmed, and (d) as a Methods-section cross-check
if time allows. Is (d) worth the effort, and is our error-bar presentation
sufficient for the thesis?

**Related:** the slow observed order also limits how much brute-force refinement
helps — worth asking whether this is typical for excitable-media FDM work or a
sign we should reconsider the discretisation (e.g. the flux-form tensor diffusion).
