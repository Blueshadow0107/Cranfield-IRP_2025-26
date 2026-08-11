# CHANGELOG — shortened chapter drafts in `Report/edits/`

Date: 2026-08-03. Each file in this directory is a copy-edit of the live chapter in `Report/chapters/`, with a header comment marking it as such. The edits are almost entirely sentence-level tightening: long multi-clause sentences split or compressed, redundant qualifiers dropped, enumerations collapsed. **No `\label{}` was added, removed, or renamed in any of the four files, and the set of citation keys is byte-identical** (verified by diffing extracted labels and cite keys for each pair). **No number, parameter value, or quantitative claim changed** — all speeds, percentages, eigenvalues, firing counts, and working-point parameters are identical between versions. The one structural (non-prose) change is in `substrates.tex`: the novelty-headroom column was removed from `tab:substrate_comparison`.

## `introduction.tex` (195 -> 194 lines)

- Opening von Neumann / physical-neural-network paragraph compressed from 4 sentences to 5 shorter ones; citations unchanged.
- Premise paragraph tightened ("deliberately open premise" -> "open premise", em-dash style list).
- Chapter-roadmap paragraph: the five `\refSec` signpost sentences collapsed into two sentences; all `sec:lit_*` references retained.
- Literature-review exemplar paragraphs (`sec:lit_pnn`, `sec:lit_chemical`, BZ/Oregonator/photosensitive-BZ passages, phenomenology survey, synthesis/gap) reworded for economy; every citation key and every claim preserved.
- Aim/objectives/scope: minor rewording of two `\item` entries and the scope bullet on Barkley/Oregonator.
- Closing thesis-outline paragraph: one summarising sentence dropped ("The remainder of the thesis is organised as follows." / "recording what this project itself did with each"); all four `\refChap` references retained.

## `methods.tex` (was `methodology.tex`; 232 -> 233 lines)

- Overview: "three layers" description rewritten from noun-first to adjective-first phrasing; content identical.
- Oregonator parameter paragraph: gloss of `q` shortened (dropped "arising from the disparity of the reaction rates").
- Discretisation section: sentence restructuring only; `Delta x = 1`, `Delta t = 0.05`, subcycling rule `h <= 0.5/max(-dF/du,0)`, `-q/2` limiter all unchanged.
- **2026-08-03 (post-CHANGELOG, both versions): the clipping/de-hack narrative was removed at the user's request** (methods paragraph + results exemplar paragraph, deleted identically in `chapters/` and `edits/`). Kept in methods: the stiffness motivation (Jacobian ~ -7000 during the upstroke) and the LSODA 1e-6 validation, restated without the hack story.
- Verification paragraph restructured (Oregonator vs Barkley convergence now one sentence); the placeholder values `x%`, `y%`, `z%` were unresolved at that point — **later resolved by deleting the two placeholder tables entirely** in the de-duplication pass below.
- Protocol section: comma splice fixed ("The reason is physical, At an open junction" -> "The reason is physical. At an open junction"); windowed-readout justification kept in full.
- Four experiment descriptions: list punctuation normalised (capitalised "The/And" items -> lowercase); all measured quantities and `\refSec` cross-references retained.

## `results.tex` (255 -> 251 lines)

- Grid-convergence paragraph compressed from 6 sentences to 4; all numbers identical (4.151 -> 4.603, asymptote ~4.82, ~14% deficit, order ~0.8, 6.224/6.048/6.049, 2.8% band, 1.5% timestep, 0.7% separability).
- Clipping paragraph: removed 2026-08-03 (both versions) — see note under `methods.tex`.
- Channel-wire, bandwidth, logic, and anisotropy discussions tightened; all quantitative claims (r = 2/4/8 deviations 4.6%/0.24%, 9.6%/0.34%, 10.2% vs ~14% error bar) unchanged.
- Two figure references merged into one sentence (`fig:res_logic` + `fig:res_logic_snapshots`); both refs retained.
- Synthesis section: minor compressions ("It is a slow, parallel, threshold-logic medium." folded into previous sentence).

## `substrates.tex` (124 -> 128 lines)

- **Structural change:** `tab:substrate_comparison` — the "Novelty headroom" column (9th column) removed from the header and all six rows; the per-row novelty entries (low / low / uncertain / moderate / low / high) are gone from the table. Caption updated to state the novelty-headroom criterion is omitted from the table and discussed in the text. Table typography changed to fit the narrower layout: `\small` -> `\scriptsize`, `\tabcolsep` 2.5pt, column spec changed from `>{\raggedright\arraybackslash}p{...}` to `>{\hspace{0pt}}p{...}` with adjusted widths.
- Text updated for consistency: "against the eight selection criteria" -> "against the selection criteria" (the in-text criterion list in the intro paragraph still enumerates all eight, including novelty headroom — intentional, since novelty is now prose-only).
- Per-family sections (acoustic, thermal, CRN/DNA, RD) and the selection rationale: sentence-level tightening only; all citation keys and the six-family list unchanged.

## Cross-chapter de-duplication pass (2026-08-03, edits/ only)

Applied after a redundancy audit of the five chapters; `chapters/` untouched.

- **methods.tex**: removed the duplicate convergence tables `tab:timestep_convergence` and `tab:grid_convergence` (cell-for-cell copies of `tab:res_convergence` in results, still holding `x%/y%/z%` placeholders — deletion resolves the placeholder issue by removal, not filling). Convergence prose compressed to a pointer at `\refSec{sec:res_verification}`; the "deficit carried openly" reporting stance kept.
- **methods.tex**: runtime-invariant *outcomes* (tripwire never fired, zero guard trips) moved to results; methods keeps the mechanism statement and the 155-substep validation detail.
- **methods.tex**: Oregonator reduction history compressed to a pointer at `\refSec{sec:lit_bz}` (was near-verbatim with the introduction).
- **methods.tex**: anisotropy experiment now refers to "the sqrt(r) eikonal law of `\refSec{sec:meth_anisotropy}`" instead of restating it.
- **results.tex**: two-kinetics-generality claim compressed to a pointer at `\refSec{sec:meth_overview}`.
- **results.tex**: OR/XOR diffraction mechanism no longer re-derived (pointer to `\refSec{sec:meth_protocol}`); sqrt(r) law no longer re-derived (pointer to `\refSec{sec:meth_anisotropy}`).
- **introduction.tex**: well-mixed CRN paragraph compressed to survey function, detail deferred to `\refSec{sec:sub_crn}`; overlap warning added to the two commented-out optional prior-work subsections (they duplicate `sec:sub_crn` / `sec:sub_acoustic` if enabled).
- **substrates.tex**: enumerated phenomenology list replaced by pointer to `\refSec{sec:res_verification}`; novelty-gap claim removed from the RD family section (it is stated in the introduction's gap section and again in the selection rationale — one substrates copy kept at line ~127).
- Verified: swapped into a scratch copy of `Report/`, compiles clean with zero undefined references.

## Things to check before swap-in

- Compile the full thesis (`cd Report && latexmk -pdf main`) with the edits swapped in and check the log for undefined references — none are expected (labels/cites identical), but the table column-spec change in `substrates.tex` is the only non-prose edit and should be eyeballed in the PDF for overfull boxes.
- ~~The methodology placeholders `x%`, `y%`, `z%`~~ — resolved: the placeholder tables and sentences were deleted in the de-duplication pass (edits/ only; they still exist in `chapters/methods.tex` until swap-in).
- Confirm you are happy that novelty headroom is now prose-only in the substrates chapter: the intro paragraph still lists it as a criterion, and the acoustic/RD sections argue it in text, but the table no longer scores it.
- `AGENTS.md` states the `edits/` versions were verified to compile; that verification was done before this review, so recompile after the actual swap.
