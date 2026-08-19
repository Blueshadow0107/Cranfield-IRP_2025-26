# Porous-reframe handoff — 2026-08-19

## What changed

The thesis has been reframed around Guo's agreed narrative: the T-junction and micro-scale junction work is the controlled 2D picture inside a larger porous chemical reactor, and the long-term goal is to extract pore-scale operators and compose them into a pore-network graph predictor.

- Archived the pre-reframe thesis files to `Report/chapters/archive/pre_porous_reframe_2026-08-19/`.
- Created a new root document: `Report/main_porous.tex`.
  - New title: *From Measured Pore-Scale Operators to Network-Level Prediction in a Structured Belousov--Zhabotinsky Reactor*.
  - New abstract written in the porous-reactor language.
- Wrote four new chapters:
  1. `introduction_porous.tex` — reframed aim/objectives around porous media and pore-network graphs.
  2. `porous_substrates.tex` — replaces the old substrates chapter; explains the porous reactor picture, why 2D controlled simulations are the micro-scale limit, the graph abstraction, and the four local operators.
  3. `methods_porous.tex` — keeps the Oregonator/solver/protocol content, adds graph-simulator construction and 3D extension sections.
  4. `results_porous.tex` — includes verification, channel, frequency, inhibition, anisotropy, graph validations (AND-NOT, router, multi-junction), XOR leak lesson, 3D inhibition, diode negative result, UQ, and training negative result.
  5. `conclusions_porous.tex` — summary of contributions, limitations, and future work.

## Compilation status

`Report/main_porous.tex` compiles cleanly with `latexmk -pdf main_porous.tex`.

- Output: `main_porous.pdf`, 36 pages.
- No undefined citations or cross-references after adding the missing bibliography entries.
- Minor overfull hboxes remain; these are cosmetic and can be fixed with wording or line breaks.

The missing citation keys (`joekar2020pore`, `blunt2017multiphase`, `meakin1991diffusion`, `kinouchi2006random`, `lopes2012collective`) were replaced with real, verifiable entries:

- `xiong2016review` — Xiong, Baychev and Jivkov, *J. Contam. Hydrol.* 192 (2016) 101-117.
- `blunt2017multiphase` — Blunt, *Multiphase Flow in Permeable Media*, CUP 2017.
- `meakin2009modeling` — Meakin and Tartakovsky, *Rev. Geophys.* 47 (2009) RG3002.
- `kinouchi2006optimal` — Kinouchi and Copelli, *Nat. Phys.* 2 (2006) 348-351.

## What is still rough

1. **Which root file is canonical?** `main.tex` is untouched. `main_porous.tex` is the new porous-framing version. You need to decide whether to keep `main.tex` as the backup and submit `main_porous.tex`, or rename `main_porous.tex` to `main.tex`.
2. **Figures are sparse.** Only two figures are currently included in the new Results chapter:
   - `pub_mms_convergence.png`
   - `pub_frequency_response_darkspot.png`
   Most results are presented as tables. If you want figure panels for the inhibition gate, router, multi-junction, XOR leak, 3D gate, diode, or UQ, they need to be generated and inserted.
3. **The old substrates chapter content is gone from the main narrative.** It has been replaced by `porous_substrates.tex`. If you still need the detailed substrate comparison (acoustic/thermal/CRN/DNA/RD) somewhere, it is in the archive but not in the new version.
4. **Negative results are in the main Results chapter.** The diode and training negative results are reported as boundary experiments. If you prefer them in a dedicated Limitations section, that is a quick move.
5. **Physical length scales and 3D framing.** The Methods chapter already has the scaling paragraph (1 t.u. ~ 50 s, 1 cell ~ 0.3 mm). The 3D inhibition gate is reported, but the statement that 2D is the controlled micro-scale limit could be strengthened with a diagram.

## Git status

All new and changed files have been committed and pushed:

```
commit 4443842
Porous-reactor reframe of thesis and diode pilot results
```

The diode pilot script and outputs (`Analysis/rd_diode_pilot.py` plus JSON/PNG figures) were also committed because they belong to the new boundary-experiment narrative.

## Recommended next steps

1. Decide on the canonical root file.
2. Add any missing figures for the graph-validation and 3D sections.
3. Read through `main_porous.pdf` and flag any sentences that still sound too AI-formal or hedged.
4. Check whether the conclusions need a stronger closing paragraph on the porous-reactor vision.
5. Run a final pass for em dashes and banned fillers before submission.

## Files to look at

- `Report/main_porous.tex`
- `Report/chapters/introduction_porous.tex`
- `Report/chapters/porous_substrates.tex`
- `Report/chapters/methods_porous.tex`
- `Report/chapters/results_porous.tex`
- `Report/chapters/conclusions_porous.tex`
- `Report/references.bib`
- `Report/chapters/archive/pre_porous_reframe_2026-08-19/`
