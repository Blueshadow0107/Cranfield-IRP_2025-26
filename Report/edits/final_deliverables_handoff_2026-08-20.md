# Final deliverables handoff — 2026-08-20

Both the thesis and the poster are built, compiled, and pushed. What follows is the state you need to review before submission tomorrow night.

## Thesis

**Root file:** `Report/main_porous.tex`

**Current PDF:** `Report/main_porous.pdf` (generated locally, not tracked)

**What changed since the reframe:**
- 10 figure environments added across Methods and Results.
- Latest PDE numbers verified against JSON outputs.
- Prose polished; em dashes and obvious AI filler removed.
- Compiles cleanly with no undefined citations or references.

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

**Key numbers in the text:**
- 2D channel speed: 6.46 cells/t.u.
- 3D channel speed: 6.62 cells/t.u.
- Refractory period: 3.0 t.u.; max 1:1 following rate: 0.167 t.u.^-1.
- Inhibition gate separation: >200x (2D), 16.8x (3D).
- Multi-junction separations: O1 182x, O2 192x.

**Still to decide:**
- `main_porous.tex` is the current root. If your submission system expects `main.tex`, either rename `main_porous.tex` to `main.tex` or set `main_porous.tex` as the submitted file.
- Do one personal read-through for voice and any last missing citations.

## Poster

**PowerPoint version:** `Presentations/MScCFD_Poster.pptx`
- Built by populating your Cranfield template.
- 3-column landscape layout, 7 figures inserted.

**LaTeX version:** `Presentations/poster.tex`
- A1 landscape beamerposter.
- Compiled PDF: `Presentations/poster.pdf`.

**Which poster to use:**
- Use `poster.pdf` if you need a print-ready PDF now (it compiled cleanly).
- Use `MScCFD_Poster.pptx` if you want to edit text or move blocks in PowerPoint.
- The PPTX was not converted to PDF because LibreOffice is not installed and I do not have sudo access on this machine.

## Git

Latest commits on `main`:
- `16fea74` — Insert figures and polish porous-reframed thesis chapters
- `cfec365` — Add MSc CFD poster: populated PPTX with porous-reactor BZ content and figures
- `0a520ab` — Add LaTeX poster source and original PPT template

All pushed to `origin main`.

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
