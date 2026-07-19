# Porous acoustic solver validation ladder

Date: 2026-06-14
Status: working plan

This document lists the stepped tests we will run before attempting the full Hughes-style vowel classifier. Each step produces a checkable result and report-ready evidence.

---

## Step 0 — Solver sanity checks

Verify the volume-velocity FDTD solver is numerically and physically correct.

| Test | What to check | Expected result |
|------|---------------|-----------------|
| Free-air pulse transit | Inject a short pressure pulse on the left, record arrival on the right | Arrival time matches `c0 = 343 m/s` |
| Hard-wall reflection | Place a rigid wall (`phi` very low or hard boundary) and send a pulse | Reflected pulse has same sign as incident (pressure doubling at rigid wall) |
| Pressure-release reflection | Place a pressure-release boundary (`p = 0`) | Reflected pulse has inverted sign |
| Energy conservation | Lossless free air, no absorbing boundaries on all sides or with periodic BCs | Total acoustic energy stays constant |
| CFL stability | Run with decreasing CFL factor | Solution converges, does not blow up |

Goal: be confident the solver propagates waves correctly before adding material complexity.

---

## Step 1 — Uniform porous medium

Run a plane wave through a slab of uniform porosity `phi` and optional flow resistivity `sigma`.

| Test | What to check | Expected result |
|------|---------------|-----------------|
| Effective sound speed | Measure pulse arrival time through `phi = 0.5` slab | `c_eff = c0 / sqrt(phi * alpha_inf)` |
| Impedance contrast | Measure reflected/transmitted amplitude at a porosity step | Matches 1D impedance formula |
| Damping scaling | Add `sigma` and measure amplitude decay | Decay rate scales with `sigma` |

Goal: confirm that `phi` and `sigma` produce the expected effective acoustic properties.

---

## Step 2 — Porosity Bragg grating

Build a periodic stack of high/low porosity layers.

| Test | What to check | Expected result |
|------|---------------|-----------------|
| Stop-band frequency | Sweep frequency through a 1D or 2D porosity grating | Dip in transmission near the Bragg frequency |
| Period scaling | Change grating period | Stop band shifts inversely with period |
| Contrast scaling | Change `phi_low` / `phi_high` | Deeper stop band with stronger contrast |

Goal: show that porosity patterns can produce frequency-selective interference.

---

## Step 3 — Single-frequency routing

Fix one frequency and hand-design a porosity map that routes the wave to a chosen probe.

| Test | What to check | Expected result |
|------|---------------|-----------------|
| Upper-probe routing | Design a `phi(x,y)` map that steers a 4 kHz wave to `y = 3Ly/4` | Highest energy at the target probe |
| Centre-probe routing | Same for `y = Ly/2` | Highest energy at the target probe |
| Lower-probe routing | Same for `y = Ly/4` | Highest energy at the target probe |

Goal: prove the solver can do spatial interference and probe-selective energy deposition.

---

## Step 4 — Broadband pulse readout

Inject a broadband pulse or short tone burst and record time-integrated squared pressure at the three probes.

| Test | What to check | Expected result |
|------|---------------|-----------------|
| Readout stability | Repeat the same input twice | Same integrated energies |
| Readout sensitivity | Small change in `phi` map changes probe energies | Measurable change in `E1, E2, E3` |
| Integration window | Vary `T_int` | Stable results once window covers the wave arrival |

Goal: establish the readout feature vector and integration window for the classifier.

---

## Step 5 — Tiny classification demo

Try a 2-class or 3-class vowel problem with a small dataset before the full 5-fold CV.

| Test | What to check | Expected result |
|------|---------------|-----------------|
| 2-class vowels | Train `phi` map + linear readout on two vowels | Better-than-chance accuracy |
| 3-class vowels | Add a third vowel | Some class separation |
| Random vs trained | Compare trained design to random `phi` map | Trained design performs better |

Goal: confirm that the pipeline (FDTD + readout + optimiser) can learn anything at all.

---

## Step 6 — Full Hughes-style vowel classifier

3 vowel classes, full dataset, 5-fold cross-validation, trainable `phi` and `sigma` maps, trainable linear readout.

| Test | What to check | Expected result |
|------|---------------|-----------------|
| Baseline accuracy | Random `phi` + trained readout | Chance-level accuracy (~33%) |
| Trained `phi` only | Optimise `phi`, fixed `sigma = 0` | Accuracy above baseline |
| Trained `phi` + `sigma` | Optimise both | Higher accuracy than `phi` alone, demonstrating extra design freedom |
| 5-fold CV | Repeat training on five folds | Stable mean test accuracy |

Goal: reproduce and extend the Hughes result in a porous medium.

---

## Current focus

Start with **Step 0** solver sanity checks.
