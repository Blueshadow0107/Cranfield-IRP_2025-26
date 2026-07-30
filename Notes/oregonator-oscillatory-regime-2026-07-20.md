# Oregonator baseline is oscillatory, not excitable — finding (2026-07-20)

## Summary

The "validated" Oregonator baseline used in `oregonator_bz_demo.py` /
`oregonator_timing_sweep.py` (Tyson–Fife scaling, `eps=0.05, q=0.002,
f=1.4, phi=0`) is **not an excitable medium**. Its spatially homogeneous
rest state is an **unstable node**; the medium is a relaxation oscillator.
Any region that has been perturbed by a wave re-fires spontaneously
(period ~47 steps homogeneous, ~70–90 steps in confined geometry). The
rest state only *appears* stable in the demos because the initial
condition `u = v = 0` is an exactly invariant point of the discrete map
and is never perturbed away from the wave paths.

## Evidence

- Fixed point on the `v = u` nullcline: `u* = v* = 0.0116`.
  Numerical Jacobian eigenvalues: **+10.84, +0.67** (both real, positive)
  → unstable node.
- Homogeneous single-cell integration (no diffusion) from `u = 1e-6`
  fires **86 times in 4000 steps** (period ≈ 46.5 steps ≈ 2.3 t.u.).
- Spatial test (`rd_core.py`, 24-wide channel, single 30-step port hold,
  port afterwards clamped to `(0,0)`): **13–15 pulses** cross a downstream
  probe within 1200–1400 steps. Space-time plot shows pulses born at the
  port end every ~68 steps, entraining the whole channel. This is why the
  reference demo's central disk produces *target waves* (repeated rings)
  rather than a single expanding ring — that is oscillatory-medium
  behaviour, not excitable-medium behaviour.

## Stability scan (fixed-point max Re(eigenvalue), f given)

| f   | eps=0.05 | 0.2  | 0.5  | 0.7   | 1.0   |
|-----|----------|------|------|-------|-------|
| 1.4 | +10.8    | +1.1 | +0.13| −0.05 | −0.19 |
| 1.2 | +13.9    | +2.3 | +0.26| +0.04 | −0.12 |

The Hopf boundary at f=1.4 lies near eps ≈ 0.63. **However**, for
eps ≳ 0.6 the medium no longer supports propagation at all (tested
eps = 0.6–1.0 with Du = 1–8 and sustained supercritical stimuli: launched
waves die immediately). The model passes directly from
*oscillatory-propagating* to *subexcitable* (stable rest, no propagation)
with **no excitable-propagating window** at these scalings.

Light suppression `phi` pushes the wrong way (effectively increases
`f·v`): phi ≤ 0.03 → everything still transmits down to W=2; phi ≥ 0.04
→ global propagation failure. The excitability range in which a
geometric wave-block could occur is essentially measure-zero.

## Consequences for the RD test battery (`rd_transfer_*.py`)

- **Leading-edge measurements on the Oregonator baseline remain valid**:
  first-crossing delays, front speeds, anisotropy ratios (the outermost
  front is unaffected by the trailing self-fired train).
- **Single-pulse protocols are impossible** on the baseline: two-pulse
  refractory, pulse-train frequency response, and collision-gate truth
  tables are all polluted by self-fired trains.
- Tests 2 and 3 (and the primary anisotropy runs) therefore use the
  project's **Barkley excitable model** (`barkley_bz_demo.py`:
  `a=0.75, b=0.01, eps=0.02, Du=1, Dv=0`), which has a stable rest state
  and a genuine threshold. Verified: a single port hold produces exactly
  one propagating pulse; two-pulse refractory ≈ 60–75 steps.
- `rd_core.RDSubstrate` has a `kinetics={'oregonator','barkley'}` switch.

## Additional findings (same session)

- **No geometric wave block** on the Oregonator baseline: channels
  transmit at full planar speed (0.3596 cells/step ≈ 7.18 cells/t.u.)
  down to **W = 1 cell**. Eikonal estimate: `kappa* = c0/Du ≈ 7.2
  cells^-1` → critical channel width ~0.3 cells, far below grid
  resolution. Block cannot be observed at DX=1 for these kinetics.
- The port-clamp trick (hold port at firing values during the pulse
  window, at `(0,0)` otherwise) is necessary but NOT sufficient for clean
  single pulses in the oscillatory medium — the re-firing originates in
  the woken bulk downstream, not in the port cells.
- Ports must be flush to the domain edge; an un-clamped pocket between
  the wall end and the port acts as an extra pacemaker.

## Collision-gate geometry findings (same session)

Systematic attempts at the naive annihilation-XOR (two inputs head-on
into a common output; singles transmit, coincident pair annihilates),
all on the excitable Barkley medium with clean single pulses:

1. **T-junction** (inputs head-on, output branch at the collision
   point): (1,1) TRANSMITS at full amplitude — the collision cusp at
   the output mouth nucleates a wave up the branch.
2. **Widened collision chamber** (48×48): the collision inside the
   chamber still fires the output; in open regions each front diffracts
   around the other — annihilation only consumes the head-on parts.
3. **Constricted output gap** (8-cell gap at the branch base): lone
   waves pass at full amplitude; no amplitude discrimination (this
   medium's propagation is binary — see point 1 above).
4. **Y-junction (±45° arms)**: (1,1) transmits; the contact line of the
   two fronts advances along the bisector (into the output) at
   c/sin(alpha) > c — a self-feeding cusp.
5. **X-junction**: lone waves diffract into ALL four arms — junctions
   compute OR.

**Conclusion:** any open junction diffracts an incoming wave into every
connected channel (OR behaviour), and a head-on collision — though it
annihilates cleanly in a plain channel — cannot be read out by any
channel touching the collision zone, because the collision cusp fires
it. This matches the BZ literature, where XOR is COMPOSED from
AND-NOT primitives rather than implemented as a single junction
(Steinbock, Kettunen & Showalter, JPC 1996; Tóth & Showalter 1995).

**Working gate implemented (TEST 3):** inhibition gate `A AND (NOT B)`:
- A travels a horizontal channel; B joins from a vertical branch;
  B's left-going fragment collides HEAD-ON with A's wave inside A's
  arm (no readout at the collision point -> clean annihilation);
- output probe far downstream with a WINDOWED readout around A's
  expected arrival (B's right-going fragment arrives earlier, outside
  the window);
- measured: (0,0)=0, (1,0)=0.997, (0,1)=0.000, (1,1)=0.000;
  inhibition window: B fired within ~104 steps of A blocks transmission.

