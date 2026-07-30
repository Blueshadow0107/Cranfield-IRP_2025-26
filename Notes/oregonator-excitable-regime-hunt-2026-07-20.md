# Oregonator excitable-regime hunt + rd_core de-hack — 2026-07-20

**Purpose.** Two coupled problems: (1) the `np.clip` calls baked into the
Oregonator/Barkley stepping in `Analysis/rd_core.py` were load-bearing —
they silently reshaped the dynamics (e.g. turning one-shot excitability
into perpetual oscillation); (2) at the literature baseline
(f=1.4, eps=0.05, q=0.002, phi=0) the Oregonator rest state is an
*unstable* node, so the medium self-oscillates and can never wait quietly
for input pulses. This note documents the de-hacked solver and a joint
(f, eps, phi) scan for a genuinely excitable regime, motivated by the
photosensitive-BZ experimental protocol where background illumination
holds the medium excitable.

## Part 1 — de-hacking rd_core.py

Both reference demos (`oregonator_bz_demo.py`, `barkley_bz_demo.py`) clip
u and v after every explicit Euler step; neither uses a semi-implicit
trick. The clip is not cosmetic: with the clip the Oregonator baseline
fires 1741 times in 4000 t.u.; without it (LSODA reference) the same
point kinetics is a relaxation oscillator with period ~5.1 t.u.
(~780 firings) — a completely different trajectory. (Note: an earlier
estimate of "1 firing unclipped" could NOT be reproduced; the stiff
LSODA reference self-fires with period ~5.1 t.u., consistent with the
verified unstable-node rest state, eigenvalues +10.84/+0.67. The
qualitative conclusion — the clip changes the dynamics — is unchanged.)

Removing the clips exposes the numerical problem the clips were hiding:
the kinetics are far too stiff for explicit Euler at dt=0.05.

- Barkley (eps=0.02): dt/eps = 2.5 > 2 — explicit Euler on the u-branch
  overshoots u=1 and diverges. Unclipped M=1 blows up immediately.
- Oregonator: the (u-q)/(u+q) switch gives a pointwise Jacobian
  dF/du = (1/eps)[1 - 2u - (fv+phi)*2q/(u+q)^2] reaching ~ -7000 on the
  recovery manifold (v~1, u~q) at eps=0.05. Fixed subcycling M=2 blows
  up within 8 steps; even M=5 is insufficient. Worse, a discrete
  overshoot can carry u below the pole at u = -q, where the kinetics
  diverge.

**What was implemented instead** (`Analysis/rd_core.py`, see module
docstring):

1. Operator splitting: one explicit diffusion step, then adaptive
   reaction subcycling over dt. Substep h is chosen proactively each
   substep from the current state:
   - stability: h <= 0.5 / max(-dF/du, 0) (analytic pointwise Jacobian)
   - accuracy:  h <= 0.1 / max|du/dt|
   - domain (Oregonator): h limited so no substep can carry u below
     -q/2 (step-size control, NOT clipping — the state is never altered)
   Wall cells are excluded from the maxima. No extra Laplacian work.
2. A non-intrusive guard: NaN/Inf or |u|,|v| > 10 raises
   FloatingPointError with the step number. Nothing is silently reshaped.

**Validation of the de-hacked core**

- Point kinetics vs scipy LSODA (rtol=1e-9..1e-10): Barkley matches to
  ~1e-6 over 50 t.u.; Oregonator tracks the relaxation-oscillation
  firing times within ~2% over 60 t.u. (11 firings, slight phase drift,
  expected at first order with dt=0.05); agreement ~1e-6 on the slow
  phases.
- dt convergence of the 2D Barkley free-medium pulse speed:
  4.151 / 4.141 / 4.162 cells/t.u. at dt = 0.05 / 0.02 / 0.01 —
  converged.

**Before/after (Test 1 + Test 4a re-runs, `Analysis/rd_dehack_verify.py`;
outputs `figures/*_dehacked.json/png`)**

| quantity | old clipped core | de-hacked core | verdict |
|---|---|---|---|
| Test 1 Barkley free-medium speed | 3.446 cells/t.u. | 4.151 cells/t.u. | +20.4% — outside 5%, see below |
| Test 4a r=1 c_par=c_perp | 3.431 | 4.135 | +20.5% |
| Test 4a r=2 c_par | 4.914 | 6.117 | +24.5% |
| Test 4a r=2 c_perp | 3.429 | 4.135 | +20.6% |
| Test 4a r=2 ratio vs sqrt(2) | 1.433 (1.35% off) | 1.479 (4.61% off) | ratio law survives |

The old absolute speeds do NOT reproduce within 5% — and they should
not: they were produced by unstable coarse explicit Euler (dt/eps=2.5)
patched by the clip, which underestimates the true pulse speed by ~17%.
Evidence the de-hacked value is the correct one: it is dt-converged
(4.14–4.16), plain unclipped Euler at fine dt=0.01 (no clips needed,
dt/eps=0.5) gives 4.11, and the point kinetics match LSODA. The physics
of the old tests survives: channels transmit cleanly (W=16, W=8 speeds
equal the free-medium speed) and the sqrt(r) anisotropy ratio holds
within 5%. **Caveat for the thesis: all pre-de-hack absolute Oregonator/
Barkley speeds carry a ~15–20% numerical error; ratios and block/no-block
phenomenology are unaffected.**

## Part 2 — (f, eps, phi) regime scan (point kinetics)

Method (`Analysis/oregonator_regime_hunt.py`): grid f in [0.5,3.0] (25),
eps in [0.005,0.2] (25, log), phi in [0,0.08] (17). Per point: smallest
positive fixed point (brentq on the u-nullcline with v=u; reproduces
u*=0.01161 and eigenvalues +10.84/+0.67 at the baseline), 2x2 Jacobian
eigenvalues, then classification:
- OSCILLATORY: max Re(eig) > 0, or perturbed trajectory never returns;
- STABLE-EXCITABLE: stable AND pulse (u_max>0.5) with return to rest
  after a u0=0.2 kick (u0=0.5 tried if 0.2 fails -> "high threshold");
- SUBEXCITABLE: stable, no pulse for either kick.
Pulse tests integrated unclipped with LSODA over 500 t.u.

**Result (`figures/oregonator_regime_map.png`, slices phi=0/0.02/0.04/0.06):**

- phi = 0: oscillatory for f <~ 2.3 across the WHOLE eps range — there is
  no eps-only excitable regime at these f, consistent with the earlier
  eps scan (oscillatory -> subexcitable without an excitable window).
- **The excitable island exists and it is exactly where the experimental
  story says it should be: at f = 1.4375 (nearest grid point to the
  literature f=1.4), phi >= 0.01 makes the rest state stable-excitable
  across the ENTIRE eps range scanned (0.005–0.2).** phi = 0.005 also
  works at eps = 0.05. Background illumination creates the excitable
  regime at the literature parameters — no eps tuning needed.
- Increasing phi further shifts the rest state down (u*: 0.0031 at
  phi=0.01 -> 0.0022 at phi=0.04) and weakens excitability; the
  high-f/small-eps corner is subexcitable; a high-threshold excitable
  band sits at large eps (>= 0.15).

## Part 3 — candidate propagation validation (de-hacked rd_core)

Channel: 300x40, width 12, walls; port slab x in [0,18); probes at x=80
and x=180 (p2 is 162 cells from the port edge = the >=100-cell
criterion); channel initialised AT the rest state (u*, v*=u*) — see
pitfall below. Candidates at f=1.4375, q=0.002 (authoritative numbers in
`figures/oregonator_regime_candidates.json`):

| f | eps | phi | propagates >=100 cells? | speed (cells/t.u.) | pulses at p1/p2 | quiet control? |
|---|---|---|---|---|---|---|
| 1.4375 | 0.0501 | 0.010 | YES | 6.21 | 1 / 1 | yes (u stays at u*=0.00308) |
| 1.4375 | 0.0199 | 0.010 | YES | 11.70 | 1 / 1 | yes |
| 1.4375 | 0.0501 | 0.040 | NO — pulse dies before x=80 | — | 0 / 0 | yes |

Two validated candidates, both at the literature f and held excitable by
background light alone:

1. **f=1.4375, eps=0.0501, phi=0.010** — the literature baseline plus
   light. Single clean pulse, speed 6.21 cells/t.u., travels >160 cells,
   medium returns to rest and stays quiet (exactly one crossing at each
   probe — no wake re-firing, unlike the old phi=0 oscillator).
2. **f=1.4375, eps=0.0199, phi=0.010** — smaller-eps variant. Speed 11.70
   cells/t.u. (faster kinetics -> faster wave), same clean single-pulse
   behaviour. Probe peak 0.71 vs 0.54 at eps=0.05 (probe means are
   diluted by the r=2 probe disk; channel max u ~0.7-0.9).

Third candidate is an honest negative (see below). A phi bracket at
f=1.4375, eps=0.0501 (run afterwards, in the candidates JSON under
`phi_failure_bracket`) localises the propagation-failure boundary:

| phi | outcome at eps=0.0501, f=1.4375 |
|---|---|
| 0.010 | clean pulse, 6.21 cells/t.u., >160 cells |
| 0.020 | pulse travels ~160 cells but ATTENUATES (p2 peak 0.428 < 0.5) — marginal |
| 0.030 | pulse dies within ~20 cells |
| 0.040 | pulse dies within ~20 cells |

So at the literature (f, eps) the usable light-holding window is roughly
phi in (0.005, 0.02): below ~0.005 the rest state is unstable
(oscillatory), above ~0.02 excitability is too weak to sustain a
propagating pulse. This is the quantitative version of the experimental
story: the background illumination level is a tuning knob with a usable
window, not a binary switch.

## Honest negatives / pitfalls

- The earlier "1 firing in 4000 t.u. unclipped" point-kinetics estimate
  is wrong; LSODA gives a period-5.1 t.u. relaxation oscillation. The
  clip-changes-dynamics conclusion is unaffected.
- Old clipped absolute pulse speeds are ~15–20% low (numerics, not
  physics). Ratios (sqrt(r), block thresholds) are unaffected.
- **Point-excitable does NOT imply propagation.** f=1.4375, eps=0.0501,
  phi=0.040 fires a clean pulse in the point kinetics (u_max>0.5, return
  to rest) but the spatial pulse dies within ~20 cells of the port:
  higher phi weakens excitability below the propagation threshold.
  Candidate selection for computing experiments must use spatial
  propagation, not the point-kinetics map alone.
- **Initial-condition pitfall:** du/dt(u=0,v=0) = phi/eps > 0, so a
  channel initialised at (0,0) is NOT at rest; at eps=0.0199 the
  homogeneous transient toward u* crosses the excitation threshold and
  the whole channel ignites synchronously at t~0.4 (both probes
  "crossed" at t=0.35 in the first propagation run — physically
  impossible by diffusion). All validation runs must initialise at
  (u*, v*=u*).
- A RuntimeWarning (harmless, division context) occasionally surfaces
  from the h_dom step-limit line in `_react_adaptive` when react == 0 on
  some cells; the computed h is unaffected (those cells are excluded by
  the falling mask).
