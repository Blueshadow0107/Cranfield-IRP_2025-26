# Literature-grounded substrate trade-off evaluation

Date: 2026-07-20
Purpose: Independent, evidence-first comparison of candidate physical computing substrates for the IRP. Unlike the earlier internal comparison notes (which summarised our own pivot history), this evaluation was produced by five independent panel reports instructed to use ONLY external verifiable literature and to grade honestly. Raw panel reports archived in the session log; this note is the synthesis.

Method: each substrate family was graded A–F on eight fixed criteria by a separate researcher with no access to project conclusions. Citations were verified (title/venue/year) by the panellists; DOIs are collected in the companion bib for Zotero.

## Grades at a glance

| Criterion | Porous-acoustic | Free-acoustic / metamaterial | Thermal / optothermal | Well-mixed CRN (DNA/enzyme) | Reaction-diffusion (BZ) |
|---|---|---|---|---|---|
| Continuity & parallelism | A− | A− | A− | D | A− |
| Trainability | C+ | B+ | B | C+ | **D** |
| Nonlinearity | D+ | D+ | B | A− | **A** |
| Speed | B | C | D | D | **E** |
| Input / output | B | B+ | B | B+ | B |
| Simulation tractability | A− | A | A | A | A |
| Experimental realisability | B− | B | C+ | B− | B |
| Novelty headroom | B | C+ | B+ | C+ | C+ |

## Per-family verdicts

### Porous-acoustic (Biot / JCA)
- Strongest demonstration: Weng et al., meta-neural-network, Nat. Commun. 11:6309 (2020) — trained, fabricated, passive acoustic classifier (but NOT porous media).
- Fatal weakness: the JCA framework exists to model *dissipation*; the most natural "weight" (flow resistivity) is exactly the knob that destroys the coherent interference computation needs. Strictly linear in the valid regime.
- Verdict: not recommended as primary substrate. Novel, but "novel for a reason the examiners will spot." The entire porous-acoustics community optimises for absorption — the orthogonal goal.

### Free-acoustic / metamaterial
- Strongest demonstration: Weng et al. 2020 (trained passive acoustic classifier, experimental); Momeni et al., Science 382:1297 (2023) — in-situ trained acoustic PNN.
- Fatal weakness: linearity — a passive acoustic network is a single linear map regardless of depth (McMahon, Nat. Phys. 2024); nonlinearity must be imported. ms-scale latency.
- Verdict: ideal *simulation* platform, but the two headline results an MSc would aim at are already published in Science/Nature venues; 2024 roadmap signals a maturing field. Baseline, not contribution.

### Thermal / optothermal
- Strongest demonstration: Romano et al., "Analog Computing with Heat" (2025, arXiv:2503.22603) — topology-optimised heat-conduction MVM, >99% accuracy; VO2 neuristor arrays (Nat. Commun. 15, 2024) for experimental nonlinearity.
- Fatal weakness: diffusive slowness + crosstalk. Thermal time constant scales as L^2/alpha — speed scales *against* you as the device grows; macroscopic thermal logic never realised experimentally despite 20 years of phononics theory (Li et al., Rev. Mod. Phys. 84:1045, 2012).
- Verdict: best simulation-to-effort ratio and genuinely uncrowded, but the physics fights the premise; no experimental anchor. Matches our own thermoviscous null result.

### Well-mixed CRN (DNA strand displacement / enzymatic)
- Strongest demonstration: Cherry & Qian, Nature 559:370 (2018) — test-tube winner-take-all network recognising 100-bit patterns.
- Fatal weakness: weights are synthesised, not adjustable — the medium is programmed, not trained; minutes-to-hours per inference; single-use destructive readout. And it fails the continuity premise *by construction* (lumped ODE, no space).
- Verdict: safest technically, weakest fit to the Project 11 premise. Trainable-CRN simulation niche already crowded (Vasic et al. PNAS 2022; Dack et al. arXiv:2406.03456).

### Reaction-diffusion / excitable media (BZ)
- Strongest demonstration: Steinbock, Toth & Showalter, Science 267:868 (1995) — BZ medium solving optimal-path through a labyrinth in one parallel wave sweep. Modern revival: Sharma et al., Nat. Commun. 15 (2024) — programmable BZ cell array.
- Fatal weakness: speed (~0.1–1 mm/s wave speed; seconds-to-minutes per gate) and non-cascadability — waves annihilate, reagents deplete, gates cannot chain without electronic transduction.
- Verdict: recommend WITH a scope correction. Not "build a practical chemical computer" (speed kills that on page one). The defensible novel framing: treat geometry and the light-suppression field phi(x,y) as a *trainable weight space* and demonstrate optimisation (not hand-design) of an input–output map — directly attacking the family's weakest grade (Trainability D), which is the literature's open gap.

## Synthesis: what this means for the thesis

1. **No substrate scores well on everything.** Every family has a fatal weakness; the question is which weakness you can turn into a contribution.
2. **RD is the only family with an A-grade native nonlinearity AND A-grade continuity AND cheap validated simulation AND £100-scale experimental reality.** Its two bad grades are Speed (E) and Trainability (D).
3. **Speed (E) is unfixable and must be conceded** — it goes in Limitations. The thesis never claims practical throughput; it claims a characterisation of what the substrate natively computes.
4. **Trainability (D) is the opportunity.** All five panels independently converged on the same gap: physical substrates are hand-designed, rarely trained. For RD specifically, nobody has trained geometry or an illumination field. Our black-box transfer-function characterisation + trained-geometry programme is aimed exactly at this gap.
5. **The thermal null result is corroborated externally** (speed/crosstalk scaling), and the acoustic novelty worry is corroborated (crowded, roadmap-mature). Both pivots are defensible on external evidence, not just our internal narrative.
6. For the lit review's substrate-selection table: this note supersedes the internal comparison; grades and citations here are the citable version.

## Key verified references (per family; bib entries in companion file for Zotero)

Porous/free acoustic: Silva et al. Science 2014 (metamaterial math ops); Hughes et al. Sci. Adv. 2019 (wave-as-RNN); Weng et al. Nat. Commun. 2020 (meta-neural-network); Momeni et al. Science 2023 (in-situ PNN training); Wright et al. Nature 2022 (deep PNN); McMahon Nat. Phys. 2024 (linearity critique); Zangeneh-Nejad et al. Nat. Rev. Mater. 2021 (review); 2024 Acoustic Metamaterials Roadmap, J. Phys. D 58:433001 (2025); Johnson, Koplik & Dashen JFM 1987; Champoux & Allard JAP 1991; Lin et al. Science 2018 (D2NN); Li et al. Nat. Commun. 2014 (granular acoustic switch); Bringuier et al. JASA 2011 (acoustic gates); Zhang et al. APL 2015 (phononic logic).

Thermal: Romano et al. 2025 arXiv:2503.22603 (heat MVM); VO2 neuristor array, Nat. Commun. 15 (2024); del Valle et al. Nature 569:388 (2019); Li, Wang & Casati APL 88:143501 (2006, thermal transistor); Wang & Li PRL 99:177208 (2007, thermal logic); Li et al. Rev. Mod. Phys. 84:1045 (2012, phononics colloquium); Shen et al. Nat. Photonics 11:441 (2017, thermo-optic trained vowel chip).

Well-mixed CRN: Cherry & Qian Nature 2018; Qian & Winfree Science 2011 (seesaw/square-root); Okumura et al. Nature 2022 (enzymatic neurons); Vasic et al. PNAS 2022 (trained CRNs); Zhang & Winfree JACS 2009 (toehold kinetics); Soloveichik et al. Nat. Comput. 2008 (CRN computation); Simmel et al. Chem. Rev. 2019 (DSD review); Srinivas et al. Science 2017 (CRN oscillator); Yang et al. Nat. Rev. Chem. 2024 (DNA computing review); Dack et al. arXiv:2406.03456 (RNCRN).

Reaction-diffusion: Steinbock, Toth & Showalter Science 1995 (maze); Adamatzky & De Lacy Costello PRE 2002 (XOR gate); Toth & Showalter JCP 1995 (logic gates); De Lacy Costello & Adamatzky Chaos Solitons Fractals 2005 (collision gates); Kuhnert Nature 1986 (photochemical memory); Kuhnert, Agladze & Krinsky Nature 1989 (image processing); Field & Noyes JCP 1974 (Oregonator); Barkley Physica D 1991 (fast excitable model); Sharma et al. Nat. Commun. 2024 (programmable BZ array); Tsompanas et al. Biosystems 2021 (light-sensitive gates); Adamatzky Phil. Trans. R. Soc. B 2019 (liquid computers); Duenas-Diez & Perez-Mercader Front. Chem. 2021 (native chemical computation review); Suematsu et al. Materials 2021 (BZ wave speed); Miller Evol. Intel. 2014 (evolution-in-materio).
