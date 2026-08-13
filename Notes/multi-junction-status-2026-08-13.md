# Multi-junction shared-control gate status — 2026-08-13

## Files touched

- `Analysis/rd_multi_gate_pde.py` — geometry, port placement, spot radius, probes, truth-table logic
- `Analysis/rd_multi_gate_graph.py` — updated edge lengths to match the PDE geometry
- `Analysis/figures/rd_multi_gate_pde.json`
- `Analysis/figures/rd_multi_gate_graph.json`
- `Analysis/figures/rd_multi_gate_pde_snapshots.png`
- `Analysis/figures/rd_multi_gate_pde_series.png`

## What was changed

1. **Confirmed graph-model truth table.** `rd_multi_gate_graph.py` now uses
   `a_to_j1=10`, `a_to_j2=50`, `b1_to_j1=25`, `b2_to_j2=60`, `j1_to_o1=40`,
   `j2_to_o2` = 40 cells.  The event-driven simulator gives the expected
   shared-control selective gate:
   - `A=0`: B1/B2 pass to O1/O2.
   - `A=1`: both paths are blocked.

2. **Fixed PDE geometry.**
   - Dark-spot ports moved to the top entrance of each vertical channel:
     B1 at `(35, 115)`, B2 at `(75, 150)`.
   - A port moved right to `(25, 90)` so it reaches J1 then J2 with short
     lead times (10 and 50 cells).
   - Spot radius increased to the calibrated `r=6`.
   - Junctions spaced 40 cells apart (J1 at x=35, J2 at x=75) so a B1 pulse
     entering the shared A channel reaches J2 *after* B2 has passed.
   - Probes placed on thin horizontal strips just below each junction
     (y=49:52), where the transmitted pulse is still strong.

3. **Fixed truth-table / separation logic.**  O1 is true for patterns with
   `B1=1, A=0` (`010`, `011`); O2 is true for `B2=1, A=0` (`001`, `011`).

## Commands run

```bash
cd Analysis
../.venv/bin/python rd_multi_gate_graph.py
../.venv/bin/python rd_multi_gate_pde.py
```

## Key numerical results (PDE)

| pattern | O1 peak | O2 peak |
|---------|---------|---------|
| 000     | 0.003   | 0.003   |
| 010     | 0.561   | 0.003   |
| 001     | 0.003   | 0.593   |
| 011     | 0.561   | 0.003   |
| 100     | 0.003   | 0.003   |
| 110     | 0.003   | 0.003   |
| 101     | 0.003   | 0.003   |
| 111     | 0.003   | 0.003   |

- O1 separation (true / max false): **182x**, recognised = **True**
- O2 separation: **192x**, recognised = **False** because `011` O2 does not fire
- Control (000): max u = u* (quiet)
- B1-only arrival at O1: 10.0 t.u.
- B2-only arrival at O2: 15.35 t.u.

## Diagnosis

A-blocking works cleanly for individual data pulses: A leakage down the data
channels is negligible, and A reliably suppresses B1 and B2 when present.

The remaining failure is **B1–B2 crosstalk through the shared A channel**:
when both B1 and B2 fire, B1 reaches J1 first, enters the horizontal A
channel, and pre-conditions / blocks J2 before B2 can excite it.  In `011`,
B2 therefore fails to reach O2 even though A is absent.  This is a
network-level effect not captured by the single-junction graph primitives.

## Single most important remaining blocker

**Eliminate B1→J2 crosstalk without breaking A's ability to reach J2.**
Candidate fixes: a Y-shaped control line so the J1 and J2 branches are not
continuous, a narrow coupling throat between J1 and J2, or an absorbing /
chamber geometry that lets A pass but dissipates a B1 pulse that tries to
use the A channel as a shortcut.
