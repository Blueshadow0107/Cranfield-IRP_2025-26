# 3D BZ-PDE status — 2026-08-13

## What was checked

- `Analysis/rd_core_3d.py` — 3D Oregonator solver (operator splitting + adaptive reaction subcycling, 7-point scalar Laplacian, Neumann walls).
- `Analysis/rd_3d_transfer_channel.py` — straight 16x16 channel in a 64^3 domain.
- `Analysis/rd_3d_transfer_logic.py` — 3D two-input inhibition gate (`A AND (NOT B)`).
- Existing JSON outputs in `Analysis/figures/`.

## Findings

### 1. Straight-channel transfer is credible

| Quantity | 3D value | 2D reference |
|----------|----------|--------------|
| Pulse speed | 6.62 cells/t.u. | ~6.46 cells/t.u. (2D Oregonator, final dark-spot channel) |
| Peak at far probe | 0.70 | comparable |
| Control (no stimulus) | quiet | quiet |

The 3D scalar-diffusion extension reproduces the 2D propagation speed, which is the key prerequisite for the "micro-scale BZ network inside a porous reactor" claim.

### 2. The original 3D T-junction was a `+` intersection, not a T-junction

`rd_3d_transfer_logic.py` originally carved the vertical B channel through the whole domain:

```python
wall[TJ - W2:TJ + W2, :, CZ - W2:CZ + W2] = False
```

This gave B a downward outlet arm, so a B pulse could reach the output channel even when A was annihilated. The resulting windowed truth table was:

| inputs | window peak | comment |
|--------|-------------|---------|
| 00 | 0.003 | quiet |
| 10 | 0.689 | A passes |
| 01 | 0.271 | **B leaks to output** |
| 11 | 0.271 | **B leaks to output** |

Separation ratio: **2.5x** (recognised, but only because the decision threshold happened to sit above the leakage).

### 3. Geometry fix: close the outlet arm and scale the arms

The geometry was changed to match the 2D T-junction and give the same arm-length timing that works in 2D:

```python
wall[TJ - W2:TJ + W2, CY:, CZ - W2:CZ + W2] = False   # B stops at junction
```

Domain was enlarged to `64x48x48`, A source moved to the left (`A_SOURCE_X = 4`, A arm ~32 cells), B source placed higher/earlier (`B_SOURCE_Y = 38`, B arm ~10 cells), and the output probe moved downstream to `X_PROBE = 56` (24 cells past the junction). This makes B arrive at the junction first and gives its diffracted fragment time to leave the readout window before A reaches the probe.

### 4. Updated 3D inhibition result

Re-running `rd_3d_transfer_logic.py`:

| inputs | window peak | comment |
|--------|-------------|---------|
| 00 | 0.003 | quiet |
| 10 | 0.713 | A passes |
| 01 | 0.042 | suppressed leakage |
| 11 | 0.042 | suppressed leakage |

- A-alone arrival at probe: **8.25 t.u.**
- Separation ratio: **16.8x**
- Decision threshold: 0.378
- `logic_recognised`: **true**
- Runtime: ~154 s for 4 cases in parallel

The leakage dropped from 0.27 to 0.04 (a 6x reduction), giving a clean enough single-gate demonstration.

## Decision

Choose **(a) redesign the 3D inhibition geometry to match the 2D T-junction**, but with the caveat that it required more than just closing the outlet arm — the arm lengths and the output-probe distance also had to be scaled to match the 2D timing. Option (b) (per-geometry calibration only) is partly true operationally, but the dominant problem here was a geometry bug, not a parameter-calibration issue.

## Remaining blocker

The single most important remaining issue is that **a credible 3D multi-gate network has not been demonstrated**. Only one isolated inhibition gate exists, and even it still has ~6% leakage (false peaks 0.04 vs true 0.71). Scaling to several cascaded junctions will require:

1. Larger 3D domains or anisotropic grids so each arm can be long enough for clean windowed readout.
2. A systematic per-junction timing/calibration protocol (delays, spot radius, probe position).
3. Verification that pulses can be routed through a second junction without re-igniting the medium.

Next concrete step: run a two-stage 3D route (e.g. channel -> T-junction -> output channel -> second T-junction) to see whether the output pulse from the first gate can reliably drive the second.
