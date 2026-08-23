# Final deliverables handoff — 2026-08-20

Both the thesis and the poster are built, compiled, and pushed. What follows is the state you need to review before submission tomorrow night.

## Thesis

**Root file:** `Report/main_porous.tex`

**Current PDF:** `Report/main_porous.pdf` (generated locally, not tracked)

**What changed since the reframe:**
- 10 figure environments added across Methods and Results.
- Literature review expanded and woven into the Introduction chapter.
- Substrate-selection rationale added to the porous-reactor chapter (answers Guo's criticism).
- Uncertainty bands added to key numbers: 2D/3D speeds, refractory period, inhibition separation.
- Prose polished; em dashes and obvious AI filler removed.
- Compiles cleanly with no undefined citations or references.
- Current length: 45 pages.

**Figures now embedded:**
- `pub_regime_map.png` — regime map (Methods).
- `pub_flash_calibration.png` — dark-spot flash calibration (Methods).
- `pub_verification.png` — solver verification (Results).
- `pub_channel_transfer_darkspot.png` — channel wire operator (Results).
- `pub_tjunction_logic_darkspot.png` — inhibition gate (Results).
- `pub_anisotropic_routing.png` — anisotropic routing (Results).
- `rd_multi_gate_pde_snapshots.png` — shared-control gate (Results).
- `rd_spot_xor_protocol.png` — collision XOR baseline (Results).
- `rd_3d_transfer_logic_snapshots.png` — 3D inhibition gate (Results).
- `rd_diode_pilot_series.png` — diode negative result (Results).
- `pub_sobol_indices.png` — UQ sensitivity (Results).

**Key numbers in the text (with uncertainty bands):**
- 2D channel speed: $6.46 \pm 0.19$ cells/t.u.
- 3D channel speed: $6.62 \pm 0.20$ cells/t.u.
- Refractory period: $3.0 \pm 0.1$ t.u.; max 1:1 following rate: 0.167 t.u.^-1.
- Inhibition gate separation: $>100\times$ conservative (2D), 16.8x (3D).
- Multi-junction separations: O1 182x, O2 192x.

**Still to decide:**
- `main_porous.tex` is the current root. If your submission system expects `main.tex`, either rename `main_porous.tex` to `main.tex` or set `main_porous.tex` as the submitted file.
- Do one personal read-through for voice and any last missing citations.

## Poster

**PowerPoint version:** `Presentations/MScCFD_Poster.pptx`
- Built by populating your Cranfield template.
- 3-column landscape layout, 7 figures inserted.

**LaTeX version:** `Presentations/poster.tex`
- A0 landscape beamerposter.
- Compiled PDF: `Presentations/poster.pdf`.
- Updated with: substrate-selection block, graph-simulator MUX result, uncertainty bands on key numbers.

**Which poster to use:**
- Use `poster.pdf` if you need a print-ready PDF now (it compiled cleanly to A0).
- Use `MScCFD_Poster.pptx` if you want to edit text or move blocks in PowerPoint. Note: the PPTX is slightly stale relative to the LaTeX version.
- The PPTX was not converted to PDF because LibreOffice is not installed and I do not have sudo access on this machine.

## 2-to-1 multiplexer (2026-08-20)

A composed logic task was added to demonstrate scaling from single gates to a small network.

**Graph simulator** (`Analysis/rd_mux_graph.py`):
- Stateful switch-node model built from the measured channel speed, refractory time, and inhibition window.
- All eight S,A,B input patterns pass: output = A when S=0, output = B when S=1.
- Results saved to `Analysis/figures/rd_mux_graph.json` and `rd_mux_graph.png`.

**3D PDE validation** (`Analysis/rd_mux_3d_pde.py`):
- 80^3 Oregonator geometry with A, S, and B channels meeting at a shared output stem.
- Four one-hot patterns run in parallel across four workers; runtime 1775 s (~30 min).
- Result: **boundary/negative result**. The select pulse S fails to block A.
  - S=0, A=1, B=0: output fires (A passes), arrival 9.15 t.u. — correct.
  - S=1, A=1, B=0: output still fires, arrival 8.0 t.u. — **incorrect**, S did not inhibit A.
  - S=0, A=0, B=1: output fires (B passes), arrival 5.7 t.u. — correct.
  - S=1, A=0, B=1: output fires (B passes), arrival 5.7 t.u. — correct.
- Interpretation: the current junction geometry/timing does not produce a usable refractory inhibition in 3D. Likely causes: S path is not short enough relative to A, or the 3D wave geometry does not create a clean refractory shadow at the junction. The graph simulator still demonstrates that the composed MUX task is accessible once the gate-level 3D inhibition is fixed.
- Results saved to `Analysis/figures/rd_mux_3d_pde.json`.

## Additional operator pilots (2026-08-20)

Four quick pilots were run to see what other graph operators are available in the same substrate.

**In the current regime:**
- **Fan-out splitter** (`rd_operator_fanout.py`): a symmetric Y-junction copies a single input pulse to two output arms. Up arm arrival 15.3 t.u., down arm 16.45 t.u. This is a usable graph rule: one incoming edge triggers multiple outgoing edges with measured per-arm delays.
- **Merge / OR-with-timing** (`rd_operator_or_merge.py`): two opposing inputs converging on a vertical output stem produce an output for every tested relative delay (-6 to +6 t.u.). The peak is almost independent of delay, so the geometry acts more like a passive combiner than a true OR/AND discriminator. A different junction shape would be needed to get a sharp delay-dependent response.

**Out of regime:**
- **Asymmetric diode** (`rd_operator_diode.py`): a ramp-plus-step channel blocked the pulse in both forward and reverse directions. The result confirms that a simple asymmetric wall is not enough for robust rectification; a more sophisticated shape (angled branches, absorbing side chambers) is needed.
- **Sub-excitable wave fragment** (`rd_operator_subexcitable_pilot.py`): lowering phi0 toward the propagation threshold (0.006) extinguishes propagation; at 0.007 a full target wave still forms. No clean stable wave-fragment window was found in the scanned range with the chosen spot size, so collision-based fragment logic is not immediately accessible without a finer parameter search or a smaller spot.

These pilots are in `Analysis/rd_operator_*.py` with JSON/PNG outputs in `Analysis/figures/`.

## Git

Latest commits on `main`:
- `fb0ee93` — Update poster to A0 and add substrate selection, MUX result, and uncertainty bands
- `c578d48` — Expand intro literature review, add substrate selection rationale, and add uncertainty bands to key results
- `90bd5cb` — Add length-aware inhibition graph sim and feasible-region sweep
- `9a853e5` — Add 3D PDE 2-to-1 MUX validation (boundary result: S fails to block A)
- `bdd08e7` — Add graph-simulator 2-to-1 multiplexer

All pushed to `origin main`.

Nothing untracked remains.

## Recommended order for your final pass

1. Open `Report/main_porous.pdf` and skim each chapter for flow.
2. Decide the root file name for submission.
3. Open `Presentations/poster.pdf` and check that the font size and figure quality are acceptable for printing.
4. If you want the PPTX version instead, open `Presentations/MScCFD_Poster.pptx` and adjust.
5. Submit both files tomorrow night.

## What I would still do if there were more time

- Run a second polish pass specifically for supervisor feedback language.
- Add a dedicated porous-reactor schematic diagram (hand-drawn or TikZ) to the Introduction.
- Convert the PPTX poster to PDF via LibreOffice once you have admin rights or a Windows/Mac machine.
- Run a final `latexmk -pdf` after any last-minute text edits.
