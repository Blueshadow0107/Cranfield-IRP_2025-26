# AI Agent Usage Log — Cranfield IRP 2025-26

## Agent identity
- Tool: Kimi Code CLI (Moonshot AI)
- Session date: 2026-08-10
- Purpose: coding assistant / computational-physics pair programmer for MSc thesis

## What the AI agent did
The AI agent assisted with the following tasks during this session:

- Explored the project state and latest logs from `AGENTS.md` and the personal database.
- Generated and displayed figures (`rd_tjunc_hand_phi_maps.png`, `rd_tjunc_frequency.png`, `rd_tjunc_frequency_tflash_2d.png`) using existing Python scripts.
- Ran frequency-response characterisation of the T-junction router (`rd_tjunc_frequency.py` and variations).
- Measured wavefront spatial extent in the Oregonator T-junction.
- Tested feasibility of pulse-duration routing and frequency routing with static `phi` fields (negative results).
- Created a brainstorm document on substrate limitations (`Analysis/figures/substrate_limitations_brainstorm.md`).
- Created a parallel audit-report version of the thesis in `Report_thatdidntgetthejobdone/`:
  - Wrote `main.tex` with audit framing.
  - Wrote five chapters: introduction, substrates, RD characterisation, forward, conclusions.
  - Created a shared Matplotlib style module (`Analysis/figures/audit_style.py`).
  - Remade 10 audit-style figures (`*_audit.png`).
  - Compiled the report to `main.pdf` (29 pages).
- Wrote a prototype for trainable wall geometry (`Analysis/rd_tjunc_trainable_geometry.py`) and started a CMA-ES pilot.
- Extracted the official project description from `CFD Project List for IRP 2025-2026.pdf`.
- Aligned the audit-report introduction with the official Project 11 three-phase brief.
- Added AI-usage acknowledgment to the audit-report `main.tex`.
- Added a silicon-comparison section (`chapters/forward.tex`).
- Fixed the substrate comparison table layout (rotated, `sidewaystable`).
- Cleaned up overfull/underfull boxes so the report compiles with only minor warnings.
- Ran a trainable wall-geometry CMA-ES pilot (`Analysis/rd_tjunc_trainable_geometry.py`) for 10 min; it timed out after 50 evaluations without improving beyond the random baseline (loss ~2.23, hard errors 2/4). Result is documented as a negative result.
- Added Parrilla-Gutiérrez et al. 2020 (Nat. Commun., programmable BZ chemical computer) to the audit report bibliography and integrated it as the leading experimental benchmark.
- Added a new section in `chapters/forward.tex` comparing the project scope against the Parrilla-Gutiérrez benchmark and reframing the work as an audit of the autonomous continuous-RD limit they leave as future work.
- Added a paragraph in `chapters/substrates.tex` acknowledging the same benchmark within the RD substrate discussion.

## What the student did
The student directed all research decisions, interpreted results, chose the audit framing, and approved the parallel-report plan. All physical reasoning, project pivots, and scientific judgments are the student's own, informed by prior work documented in `AGENTS.md` and the personal database.

## Notes for transparency
- No code or text was copied from external sources without citation.
- All figures are generated from the student's own simulation data or from existing project scripts.
- The AI agent did not perform any wet-lab or physical experiments; all results are numerical.
- The parallel report reuses the existing `CUThesis2025.sty`, `references.bib`, and `CULogo.png` from `Report/`.

## Files created/modified by the AI agent in this session
- `Report_thatdidntgetthejobdone/main.tex`
- `Report_thatdidntgetthejobdone/chapters/introduction.tex`
- `Report_thatdidntgetthejobdone/chapters/substrates.tex`
- `Report_thatdidntgetthejobdone/chapters/rd_characterisation.tex`
- `Report_thatdidntgetthejobdone/chapters/forward.tex`
- `Report_thatdidntgetthejobdone/chapters/conclusions.tex`
- `Analysis/figures/audit_style.py`
- `Analysis/figures/remake_audit_figures.py`
- `Analysis/figures/rd_*_audit.png` (10 figures)
- `Analysis/figures/substrate_limitations_brainstorm.md`
- `Analysis/rd_tjunc_trainable_geometry.py`
- `Agentic/project_description.txt`
- `Agentic/ai_usage_log.md`
- `Agentic/acknowledgment_draft.md`
- `Report_thatdidntgetthejobdone/references.bib` (added `cranfield2025irplist` entry)
- `Report_thatdidntgetthejobdone/main.tex` (added `array`, `rotating` packages, AI acknowledgment, trimmed keywords)
- `Report_thatdidntgetthejobdone/chapters/substrates.tex` (updated substrate comparison table)
- `Report_thatdidntgetthejobdone/chapters/forward.tex` (added silicon comparison)
- `Report_thatdidntgetthejobdone/chapters/introduction.tex` (aligned with project brief)
- `Report_thatdidntgetthejobdone/chapters/conclusions.tex` (updated scope)
- `Report_thatdidntgetthejobdone/chapters/rd_characterisation.tex` (minor wording fixes)
