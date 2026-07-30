# Anisotropy in Excitable and Reaction-Diffusion Media — Literature Scan

**Date:** 2026-07-19
**Purpose:** Literature foundation for extending the 2D Oregonator BZ solver from a scalar
diffusion coefficient to a direction-dependent diffusion tensor `D_ij(x,y)`. Motivation:
anisotropy gives direction-dependent wave speed, which could act as a trainable substrate
property (routing, rectification, timing) for RD computing — complementary to, or instead of,
wall geometry and the light-suppression field `phi(x,y)`.
**Scope:** 5 questions — (1) canonical theory, (2) experimental anisotropic substrates,
(3) homogenization of microstructure into effective tensors, (4) anisotropy as a
design/trainable parameter in excitable-media computing (novelty check),
(5) transfer-function / signal-processing characterisation of excitable channels.
All references below were verified by web search (title/venue/year via Crossref, PubMed,
journal TOCs, or multiple independent citing reference lists). Nothing unverified is cited.

---

## TL;DR

- The **theory** is mature, driven by cardiac electrophysiology: plane-wave speed scales as
  `c ~ sqrt(D)` along the local direction, so `c_parallel / c_perp = sqrt(D_parallel / D_perp)`;
  curvature and turning behaviour follow anisotropic eikonal-curvature equations
  (Keener 1991; Tyson & Keener 1988; Colli Franzone et al. 1990).
- **Experiments** exist in BZ chemistry: printed Ru-catalyst patterns on membranes produce
  genuine diffusion anisotropy and anchor spirals (Steinbock, Kettunen & Showalter, Science
  1995); light-projected patterns route waves (Sakurai et al., Science 2002); micropatterned
  self-oscillating polymer brushes give direction control (Homma et al., Small 2017).
- **Homogenization** gives the rigorous bridge: periodic microstructure (walls, grains,
  printed dots) averages into an effective tensor `D*_ij` (Bensoussan-Lions-Papanicolaou 1978);
  for *excitable* media naive averaging can fail to predict speed and propagation failure
  (Keener, Physica D 2000) — an important caveat.
- **Novelty (Q4):** nobody appears to have used a *spatially varying diffusion tensor field*
  as the trained/optimised parameter in excitable-media computing. Diodes/rectifiers exist but
  are built from geometry or excitability asymmetry, not diffusion anisotropy. The niche is
  essentially empty at the computing end, but the *phenomena* are heavily pre-worked in the
  cardiac literature, which must be credited. Good position for us.
- **Transfer functions:** cardiac people treat tissue as a signal-processing element
  (de Lange & Kucera 2009: Bode plots of interbeat-interval transmission); chemical-computing
  people built frequency band filters (Gorecka & Gorecki 2003) and memory/delay elements
  (Motoike et al. 2001). The "channel as filter/delay line" framing is established and
  directly usable.

---

## 1. Theory: waves in anisotropic excitable media

The canonical setting is the monodomain reaction-diffusion model of cardiac tissue:

```
du/dt = div( D(x) grad(u) ) + f(u,v),   dv/dt = g(u,v)
```

with `D(x)` a symmetric positive-definite tensor whose principal axis follows the local
muscle-fibre direction. Key results:

- **Direction-dependent speed.** For a plane wave propagating in direction `n`, the speed is
  the isotropic speed evaluated at the directional diffusivity `D_n = n^T D n`, i.e.
  `c(n) ~ sqrt(D_n)`. Along vs across fibres:
  `c_parallel / c_perp = sqrt(D_parallel / D_perp)` (typically a ratio ~2-3 in ventricle).
- **Anisotropic eikonal-curvature relation.** Front normal velocity
  `c = c0(D_n) - D_n * K` (K = front curvature), with both the plane-wave speed and the
  curvature penalty becoming direction dependent. This predicts slower, more fragile
  propagation across fibres and preferential block where the front must expand against the
  slow direction.
- **Metric rescaling.** A coordinate stretch by `D^{-1/2}` maps the anisotropic problem to an
  isotropic one (at constant `D`); fronts become ellipses with axis ratio
  `sqrt(D_parallel / D_perp)` in physical space. This is the cheapest way to sanity-check a
  tensor-diffusion solver.

### Key references

**Tyson & Keener 1988** — J. J. Tyson, J. P. Keener, "Singular perturbation theory of
traveling waves in excitable media (a review)", *Physica D* 32 (1988) 327-361.
The canonical review deriving dispersion `c(omega)` and curvature `c(K)` relations for
two-variable excitable kinetics; everything anisotropic builds on these relations evaluated at
`D_n`. *Why it matters:* gives the scalar building blocks we extend with a tensor; our
Oregonator solver can reproduce `c(D)` and `c(K)` curves numerically and compare against
this theory in both principal directions.

**Keener 1991** — J. P. Keener, "An eikonal-curvature equation for action potential
propagation in myocardium", *J. Math. Biol.* 29 (1991) 629-651.
Derives the eikonal-curvature equation for anisotropic myocardium by treating `D` as a
Riemannian metric: front propagation is Huygens flow in the conductivity metric plus a
curvature correction. Canonical reference for "waves go faster along fibres, curvature costs
more across fibres". *Why it matters:* this is the reduced model against which we can
validate an anisotropic Oregonator front; also the right framework for predicting what a
spatially varying `D(x,y)` field does to arrival times (a Finsler/Fermat arrival-time problem —
i.e. routing).

**Colli Franzone, Guerri & Rovida 1990** — P. Colli Franzone, L. Guerri, S. Rovida,
"Wavefront propagation in an activation model of the anisotropic cardiac tissue: asymptotic
analysis and numerical simulations", *J. Math. Biol.* 28 (1990) 121-176.
Rigorous asymptotic analysis of front propagation with tensor diffusion, including the
rescaled-metric construction. *Why it matters:* the mathematical justification for the
`D^{-1/2}` stretching trick and for elliptical front shapes — our primary unit test.

**Cabo et al. 1994** — C. Cabo, A. M. Pertsov, W. T. Baxter, J. M. Davidenko, R. A. Gray,
J. Jalife, "Wave-front curvature as a cause of slow conduction and block in isolated cardiac
muscle", *Circ. Res.* 75 (1994) 1014-1028.
Experiment + simulation showing that where a wavefront must turn or expand (e.g. around a
pivot, at a tissue expansion), the curvature-induced slowdown is anisotropy-dependent and can
cause unidirectional block — the mechanism behind anisotropic reentry. *Why it matters:*
"anisotropic wave block at turns" — the exact phenomenon we could reproduce in the Oregonator
solver as a substrate for rectification. Note the mechanism is curvature + anisotropy, not
anisotropy alone.

**Keener & Tyson 1992** (supporting) — J. P. Keener, J. J. Tyson, "The dynamics of scroll
waves in excitable media", *SIAM Rev.* 34 (1992) 1-39. Kinematics of scroll/spiral waves;
useful background if spirals are later used as oscillators in the computing layer.

**On canonical authors:** the search confirms the names in the brief — Keener (eikonal theory,
homogenization, discrete-cell effects), Tyson (singular perturbation theory), Panfilov
(anisotropic ventricular modelling), Pertsov (experimental anisotropy, wave block),
Winfree (vortex dynamics; less on anisotropy per se). Wellner appears mainly via
Wellner-Berenfeld-Jalife-Pertsov (PNAS 2002, minimal principle for rotor filaments) rather
than a dedicated anisotropic-eikonal paper, so no Wellner first-author anisotropy paper is
cited here — do not cite one from memory.

---

## 2. Experiment: BZ and other excitable chemistry in anisotropic substrates

**Steinbock, Kettunen & Showalter 1995** — O. Steinbock, P. Kettunen, K. Showalter,
"Anisotropy and spiral organizing centers in patterned excitable media", *Science* 269 (1995)
1857-1860.
The key experimental paper for us. BZ reaction run on membranes with the Ru catalyst printed
in a regular pattern of small dots; the unprinted gaps set an effective transverse coupling,
so the medium acquires a genuine effective anisotropy (homogenized microstructure, in our
language). Waves travel faster along rows of dots; spirals pin to pattern defects, which act
as organising centres. *Why it matters:* proof that an engineered BZ substrate with
direction-dependent effective diffusion is buildable and controllable — the physical
realisation of our `D(x,y)` field. Also the experimental link to homogenization (Q3).

**Sakurai, Mihaliuk, Chirila & Showalter 2002** — T. Sakurai, E. Mihaliuk, F. Chirila,
K. Showalter, "Design and control of wave propagation patterns in excitable media",
*Science* 296 (2002) 2009-2012.
Photosensitive BZ with computer-projected light masks: excitable channels, obstacles and
junctions are printed as illumination patterns, giving designer wave routing. *Why it
matters:* the light field `phi(x,y)` they use is an *excitability* anisotropy/pattern, not a
diffusion anisotropy — this is exactly our existing `phi(x,y)` trainable field. It defines the
state of the art our tensor-diffusion idea must beat or complement.

**Homma et al. 2017** — K. Homma, T. Masuda, A. M. Akimoto, K. Nagase, K. Itoga, T. Okano,
R. Yoshida, "Fabrication of micropatterned self-oscillating polymer brush for direction
control of chemical waves", *Small* 13 (2017) 1700041.
BZ-type self-oscillating polymer brushes micropatterned into stripe geometries; chemical waves
are guided preferentially along the patterned direction. *Why it matters:* a materials route
to direction-dependent wave guidance that is diffusion/coupling-based rather than
illumination-based — a second experimental precedent for anisotropic substrates.

**Fast, Darrow, Saffitz & Kléber 1996** — V. G. Fast, B. J. Darrow, J. E. Saffitz,
A. G. Kléber, "Anisotropic activation spread in heart cell monolayers assessed by
high-resolution optical mapping. Role of tissue discontinuities", *Circ. Res.* 79 (1996)
115-127.
Patterned neonatal-rat cardiomyocyte monolayers grown as controlled anisotropic strands;
optical mapping measures `c_parallel`, `c_perp`, and block at discontinuities. *Why it
matters:* the benchmark experimental protocol for characterising an anisotropic excitable
sheet — we can mirror it in silico (pace at one end, measure directional speeds and block
thresholds).

**Bursac, Parker, Iravanian & Tung 2002** — N. Bursac, K. K. Parker, S. Iravanian, L. Tung,
"Cardiomyocyte cultures with controlled macroscopic anisotropy: a model for functional
electrophysiological studies of cardiac muscle", *Circ. Res.* 91 (2002) e45-e54.
Micropatterned growth substrates give cardiomyocyte cultures a prescribed, tunable anisotropy
ratio. *Why it matters:* demonstrates anisotropy as a *fabrication parameter* — the wet-lab
analogue of a trainable `D(x,y)`.

**Negative/absent results.** I found no BZ experiment in stretched/aligned gels or capillary
arrays that explicitly reports a diffusion-tensor measurement; Tóth & Showalter's capillary
work (below) is discrete 1D channels, and BZ gel work (e.g. Kitahata's group) is about
membranes, rings and geometry. So the experimental anisotropy literature for BZ is thin beyond
the Steinbock printed-catalyst line — worth stating in the report as an opportunity.

---

## 3. Homogenization: from microstructure to effective tensor

Classical two-scale homogenization for a periodic diffusion problem with oscillatory
coefficient `a(x/eps)`:

```
-effective equation:  div( D* grad(u0) ),   D*_ij = < a_ij + a_ik * d(chi_j)/dy_k >
```

where `chi_j` solves the cell problem on the period cell and `<.>` is the cell average.
Aligned stripes of high diffusivity in a low-diffusivity matrix yield an anisotropic `D*`
with `D*_parallel > D*_perp` — i.e. **microstructure is the physical origin of the tensor**.

**Bensoussan, Lions & Papanicolaou 1978** — A. Bensoussan, J.-L. Lions, G. Papanicolaou,
*Asymptotic Analysis for Periodic Structures*, Studies in Mathematics and its Applications 5,
North-Holland, Amsterdam, 1978.
The canonical reference: two-scale expansions, cell problem, effective tensor formula above.
*Why it matters:* gives us the rigorous statement that an array of walls/channels/dots (what
we can fabricate or simulate cheaply at coarse scale) is equivalent to a smooth tensor field
`D*(x,y)` (what we can train). It legitimates training `D(x,y)` as a stand-in for training
microgeometry.

**Xin 2000** — J. Xin, "Front propagation in heterogeneous media", *SIAM Rev.* 42 (2000)
161-230.
Review of front speeds in periodic and random media (pulsating fronts, homogenized speeds,
variational characterisations). *Why it matters:* connects homogenized diffusion to *front
speed*, which is the quantity that matters for timing/routing computation; also collects the
results on speed enhancement/depression by advection and periodic structure.

**Keener 2000** — J. P. Keener, "Homogenization and propagation in the bistable equation",
*Physica D* 136 (2000) 1-17.
Crucial caveat: standard homogenization of the diffusion coefficient fails to predict wave
speed and propagation failure for a bistable/excitable reaction term; a corrected averaging
procedure is needed. In discrete or strongly modulated media, propagation can fail
(pinning) even though the homogenized equation propagates. *Why it matters:* warns us that
simulating the homogenized `D*(x,y)` is not always equivalent to simulating the walls — for
thin channels near the propagation-failure threshold, we must check against explicit-wall
simulations. Good thesis material (a real limitation, with a citation).

(Supporting standard text: E. Sánchez-Palencia, *Non-Homogeneous Media and Vibration Theory*,
Lecture Notes in Physics 127, Springer, 1980 — cited in the homogenization literature as the
other classic; include if a second source is needed.)

---

## 4. Computing: has anyone used anisotropy as a design/trainable parameter?

**Short answer: essentially no.** The excitable-media computing literature achieves
directionality, rectification and routing through *geometry* (channels, asymmetric gaps,
chambers) or *excitability* fields (light), not through the diffusion tensor. The cardiac
literature documents anisotropy-based directional phenomena (block, reentry) as pathology,
not as computation. The combination — a spatially varying tensor `D(x,y)` as the trained
weight of an RD computer — appears to be an open niche. Searches for "anisotropic excitable
medium diode/rectifier/router/computing" return only the papers below (geometry/excitability
based) plus cardiac arrhythmia work.

### What exists (nearest neighbours)

**Agladze, Aliev, Yamaguchi & Yoshikawa 1996** — "Chemical diode", *J. Phys. Chem.* 100
(1996) 13895-13897.
The first chemical diode: two BZ excitable regions coupled through an asymmetric
geometry/excitability junction so waves pass one way and block the other. *Why it matters:*
the baseline rectifier we should compare a tensor-anisotropy rectifier against; the mechanism
(excitability/geometry asymmetry) is orthogonal to diffusion anisotropy, so our claim of a
distinct mechanism is clean.

**Sendiña-Nadal, deCastro & Gómez-Gesteira 2006** — "Kinematic description of wave
propagation through a chemical diode", *Chaos* 16 (2006) 033110.
Kinematic model of the diode: decomposition of front motion into modes (uniform-medium speed
at the leading edge vs at free ends) predicts bidirectional/unidirectional/blocking regimes
from geometry and excitability contrast. Companion to an experimental/Oregonator study in
photosensitive BZ. *Why it matters:* gives a reduced kinematic theory for diode behaviour
that we can reuse to analyse tensor-based rectifiers; also an Oregonator precedent (same
model as ours).

**Górecka, Górecki & Igarashi 2009** — J. N. Górecka, J. Górecki, Y. Igarashi, "On the
simplest chemical signal diodes constructed with an excitable medium", *Int. J. Unconv.
Comput.* 5 (2009) 129-143.
Systematic study of minimal diode geometries in excitable channels (triangular/rectangular
asymmetric junctions). *Why it matters:* defines the "device catalogue" of the structured-medium
school; any anisotropy-based device we propose should be benchmarked against these.

**Tóth & Showalter 1995** — Á. Tóth, K. Showalter, "Logic gates in excitable media",
*J. Chem. Phys.* 103 (1995) 2058-2066.
Chemical waves in networks of capillary tubes implement AND/OR gates via collision and
annihilation; gate function depends on geometry, synchronisation and tube radius vs critical
radius. *Why it matters:* canonical geometry-based logic; the comparison partner for the
planned two-input gate (Week 3 of the pivot plan).

**Steinbock, Tóth & Showalter 1995** — O. Steinbock, Á. Tóth, K. Showalter, "Navigating
complex labyrinths: optimal paths from chemical waves", *Science* 267 (1995) 868-871.
A colliding-wave construction in BZ finds shortest paths through a maze. *Why it matters:*
routing-as-computation precedent; an anisotropic `D(x,y)` field implements exactly this
"refractive index landscape" idea but with direction-dependent speed — worth framing our work
as the anisotropic generalisation.

**Adamatzky & De Lacy Costello 2002** — A. Adamatzky, B. De Lacy Costello, "Experimental
logical gates in a reaction-diffusion medium: the XOR gate and beyond", *Phys. Rev. E* 66
(2002) 046112.
Experimental XOR in geometrically structured photosensitive BZ. *Why it matters:* the
experimental benchmark for our planned gate task; again geometry-based.

**Górecki, Górecka & Igarashi 2009** — J. Górecki, J. N. Górecka, Y. Igarashi, "Information
processing with structured excitable medium", *Nat. Comput.* 8 (2009) 473-492.
Review/programmatic paper of the Warsaw school: diodes, memory, frequency filters, counters,
coincidence detectors from channel geometry + excitability contrast. *Why it matters:* the
most complete statement of what structured excitable media can compute — and it never uses
diffusion anisotropy, which is our gap statement in one citation.

**Sendiña-Nadal et al. 2001** (supporting) — I. Sendiña-Nadal, E. Mihaliuk, J. Wang,
V. Pérez-Muñuzuri, K. Showalter, "Wave propagation in subexcitable media with periodically
modulated excitability", *Phys. Rev. Lett.* 86 (2001) 1646-1649.
Periodic light modulation of photosensitive BZ: resonant/locking propagation effects.
*Why it matters:* spatially periodic substrate fields already explored — again via
excitability, not diffusion tensor.

**Recent activity (2025, noted, not formally cited):** a paper titled "A chemical diode for
neuromorphic computing: design, simulation, and experimental validation of unidirectional
signal transmission" appeared in *Eur. Phys. J. Spec. Top.* (2025), DOI
10.1140/epjs/s11734-025-01506-1 — indicates the diode/device niche is alive, but it is still
geometry-based. (Author list not verified; obtain from the DOI before citing.)

### Honest novelty statement for the report

- Taken: diodes/rectifiers from geometry (Agladze 1996; Górecka 2009), routing from light
  fields (Sakurai 2002), fixed printed anisotropy for spiral pinning (Steinbock 1995),
  anisotropy+curvature block as cardiac pathology (Cabo 1994).
- Apparently free: **training a continuous tensor field `D(x,y)` (magnitude + fibre direction)
  to realise a target input-output map in an RD computer.** The nearest conceptual overlap is
  cardiac modelling where `D(x)` is fitted from imaging data (inverse problem, not computing).
- Risk note: "trainable anisotropy" should be claimed carefully — Steinbock 1995 shows
  fabricated anisotropy is known, and inverse `D(x)` estimation is routine in cardiac
  electrophysiology. The novel part is anisotropy as the *optimised weight of a computing
  medium*, and its differentiability vs walls.

---

## 5. Transfer-function view of excitable channels

**de Lange & Kucera 2009** — E. de Lange, J. P. Kucera, "The transfer functions of cardiac
tissue during stochastic pacing", *Biophys. J.* 96 (2009) 294-311.
Derives an analytical transfer function (gain + phase in the frequency domain) for
interbeat-interval variations conducted along a cardiac strand, measured with stochastic
pacing; shows positive CV-restitution slope attenuates interval variations (low-pass-like)
while negative slope amplifies them near the alternans frequency. Validated in simulations
and cultured strands. *Why it matters:* this is literally "excitable channel as LTI-ish
signal-processing element with Bode plots" — the analysis framework we should adopt to
characterise anisotropic channels (delay, attenuation, cutoff set by refractory period).

**Górecka & Górecki 2003** — J. N. Górecka, J. Górecki, "T-shaped coincidence detector as a
band filter of chemical signal frequency", *Phys. Rev. E* 67 (2003) 067203.
A T-shaped junction of excitable channels transmits only input pulse trains within a
frequency band — an explicit chemical band-pass filter built from geometry + refractory
dynamics. *Why it matters:* proof that frequency filtering is a recognised device function in
chemical computing; an anisotropic segment gives a second, tunable filtering mechanism
(direction-dependent delay + refractory cutoff).

**Courtemanche, Glass & Keener 1993** — M. Courtemanche, L. Glass, J. P. Keener,
"Instabilities of a propagating pulse in a ring of excitable media", *Phys. Rev. Lett.* 70
(1993) 2182-2185.
A pulse circulating on a ring is a clock/delay-line whose stability (Wenckebach-like
patterns, annihilation) is set by ring length vs refractory time; derived via delay/recursion
maps. *Why it matters:* the ring is the canonical periodic channel; its input-output
(beat-to-beat) map is exactly a pulse-train transfer curve — directly reproducible in our
solver, including on anisotropic rings.

**Ito & Glass 1992** (supporting) — H. Ito, L. Glass, "Theory of reentrant excitation in a
ring of cardiac tissue", *Physica D* 56 (1992) 84-106. Full bifurcation analysis of ring
conduction; the mathematical companion to Courtemanche 1993.

**Motoike, Yoshikawa, Iguchi & Nakata 2001** — I. N. Motoike, K. Yoshikawa, Y. Iguchi,
S. Nakata, "Real-time memory on an excitable field", *Phys. Rev. E* 63 (2001) 036220.
BZ-based experimental demonstration of memory/delay using pulse interactions on an excitable
field. *Why it matters:* delay-line/memory precedent in actual BZ chemistry.

---

## 6. Synthesis: what this means for our Oregonator solver with a tensor field D(x,y)

### 6.1 Model extension

Replace `D_u laplacian(u)` by `div( D(x,y) grad(u) )` in the Oregonator activator equation
(and optionally for the inhibitor; cardiac models anisotropise only the voltage, since
chemical anisotropy of recovery species is unphysical — for BZ in a structured substrate both
species would see the same structural tensor, a point to decide explicitly).

Parametrise by a fibre-direction field `theta(x,y)` and two eigenvalues:

```
D(x,y) = D_perp * I + (D_parallel - D_perp) * f f^T,   f = (cos theta, sin theta)
```

Numerics notes (flagged for the implementation stage): a rotated tensor needs a 9-point
stencil (cross terms `D_xy`); the explicit-stability constraint tightens to
`max(eig(D)) * dt / dx^2 <= 1/4` in 2D; a sanity test is the `D^{-1/2}` coordinate stretch
recovering circular waves (Colli Franzone 1990).

### 6.2 Concrete phenomena to reproduce (each has a literature anchor)

1. **Elliptical target waves / directional speed.** Point stimulus in uniform anisotropic
   medium gives ellipse with axis ratio `sqrt(D_parallel/D_perp)`; verify
   `c_parallel/c_perp = sqrt(D_parallel/D_perp)` against Tyson-Keener dispersion. (Tyson &
   Keener 1988; Colli Franzone 1990.)
2. **Anisotropic curvature cost and block at turns.** Force a front around a bend or through
   an expansion; block occurs preferentially for propagation across the slow direction —
   the mechanism of anisotropic reentry. (Cabo 1994; Keener 1991.)
3. **Rectification at an anisotropy interface.** A slab whose fibre direction is rotated
   relative to the incoming front acts as a refractive/diffractive element; combined with
   curvature effects this should give one-way transmission without any walls — a *tensor
   diode*, to be benchmarked against the Agladze/Górecka geometry diodes. (Novel; nearest
   anchors: Agladze 1996, Sendiña-Nadal 2006.)
4. **Spiral anchoring on printed anisotropy.** Reproduce Steinbock 1995 in silico: periodic
   anisotropic microstructure pins/organises spirals. This is the homogenization check too
   (micro-dots vs homogenized tensor).
5. **Homogenization validation.** Simulate striped walls at fine scale, fit the effective
   `D*` from measured directional speeds, compare with the cell-problem prediction
   (BLP 1978) and check the propagation-failure caveat near threshold (Keener 2000).
6. **Transfer functions.** Drive a channel with stochastic interval pacing, measure the
   interbeat-interval transfer function downstream (de Lange & Kucera protocol); show the
   cutoff/delay depend on fibre orientation. Ring version: Courtemanche 1993 instabilities
   on anisotropic rings.

### 6.3 Anisotropy vs walls vs light field as trainable weights

| Property | Walls (geometry) | Light field `phi(x,y)` | Tensor `D(x,y)` |
|---|---|---|---|
| What it modulates | domain (hard BCs) | excitability/timescale | directional coupling/speed |
| Smooth/differentiable | no (binary mask) | yes | yes (eigenvalues + angle) |
| Physics precedent | Tóth-Showalter, Górecki | Sakurai 2002 | Steinbock 1995, cardiac |
| Rectification mechanism | asymmetric gaps | excitability contrast | anisotropy + curvature (new) |
| Fabricability | easy (channels) | easy (projector) | printed catalyst (Steinbock), polymer brushes (Homma) |
| Failure modes | propagation failure in narrow channels | wave death under light | Keener-2000 homogenization caveats; cross-term stencil error |

The honest positioning: `D(x,y)` is not a replacement for walls but a *smooth, everywhere-
defined* trainable field whose phenomenology (speed, refraction, block) is well predicted by
cardiac theory — which gives us strong validation targets. Its differentiability relative to
wall masks matters for gradient-based training; walls could still be the fabrication route to
an approximate `D*(x,y)` via homogenization, so the two pictures are compatible layers of one
story (microgeometry ↔ effective tensor).

### 6.4 Gaps / things not found

- No paper found that trains or optimises a diffusion-tensor field for computation.
- No BZ-in-stretched-gel or capillary-array paper with a measured diffusion tensor found.
- No dedicated Wellner/Pertsov anisotropic-eikonal theory paper verified (Pertsov appears as
  co-author on the experimental cardiac papers; the theory anchor is Keener).
- If the report needs a Wellner-camp citation, use Gray/Jalife/Pertsov-group experimental
  papers (Cabo 1994) instead.

---

## 7. BibTeX-ready entries

Vancouver-ish keys; append alphabetically to `Report/references.bib`.

```bibtex
@article{agladze1996chemical,
  author  = {Agladze, K. and Aliev, R. R. and Yamaguchi, T. and Yoshikawa, K.},
  title   = {Chemical diode},
  journal = {J Phys Chem},
  year    = {1996},
  volume  = {100},
  number  = {33},
  pages   = {13895--13897},
  doi     = {10.1021/jp9608990}
}

@article{adamatzky2002experimental,
  author  = {Adamatzky, A. and De Lacy Costello, B.},
  title   = {Experimental logical gates in a reaction-diffusion medium: the XOR gate and beyond},
  journal = {Phys Rev E},
  year    = {2002},
  volume  = {66},
  number  = {4},
  pages   = {046112},
  doi     = {10.1103/PhysRevE.66.046112}
}

@book{bensoussan1978asymptotic,
  author    = {Bensoussan, A. and Lions, J.-L. and Papanicolaou, G.},
  title     = {Asymptotic Analysis for Periodic Structures},
  series    = {Studies in Mathematics and its Applications},
  volume    = {5},
  publisher = {North-Holland},
  address   = {Amsterdam},
  year      = {1978}
}

@article{bursac2002cardiomyocyte,
  author  = {Bursac, N. and Parker, K. K. and Iravanian, S. and Tung, L.},
  title   = {Cardiomyocyte cultures with controlled macroscopic anisotropy: a model for functional electrophysiological studies of cardiac muscle},
  journal = {Circ Res},
  year    = {2002},
  volume  = {91},
  number  = {10},
  pages   = {e45--e54},
  doi     = {10.1161/01.RES.0000047530.88338.EB}
}

@article{cabo1994wavefront,
  author  = {Cabo, C. and Pertsov, A. M. and Baxter, W. T. and Davidenko, J. M. and Gray, R. A. and Jalife, J.},
  title   = {Wave-front curvature as a cause of slow conduction and block in isolated cardiac muscle},
  journal = {Circ Res},
  year    = {1994},
  volume  = {75},
  number  = {6},
  pages   = {1014--1028},
  doi     = {10.1161/01.RES.75.6.1014}
}

@article{collifranzone1990wavefront,
  author  = {Colli Franzone, P. and Guerri, L. and Rovida, S.},
  title   = {Wavefront propagation in an activation model of the anisotropic cardiac tissue: asymptotic analysis and numerical simulations},
  journal = {J Math Biol},
  year    = {1990},
  volume  = {28},
  number  = {2},
  pages   = {121--176},
  doi     = {10.1007/BF00163146}
}

@article{courtemanche1993instabilities,
  author  = {Courtemanche, M. and Glass, L. and Keener, J. P.},
  title   = {Instabilities of a propagating pulse in a ring of excitable media},
  journal = {Phys Rev Lett},
  year    = {1993},
  volume  = {70},
  number  = {14},
  pages   = {2182--2185},
  doi     = {10.1103/PhysRevLett.70.2182}
}

@article{delange2009transfer,
  author  = {de Lange, E. and Kucera, J. P.},
  title   = {The transfer functions of cardiac tissue during stochastic pacing},
  journal = {Biophys J},
  year    = {2009},
  volume  = {96},
  number  = {1},
  pages   = {294--311},
  doi     = {10.1016/j.bpj.2008.09.025}
}

@article{fast1996anisotropic,
  author  = {Fast, V. G. and Darrow, B. J. and Saffitz, J. E. and Kl{\'e}ber, A. G.},
  title   = {Anisotropic activation spread in heart cell monolayers assessed by high-resolution optical mapping. Role of tissue discontinuities},
  journal = {Circ Res},
  year    = {1996},
  volume  = {79},
  number  = {1},
  pages   = {115--127},
  doi     = {10.1161/01.RES.79.1.115}
}

@article{gorecka2003tshaped,
  author  = {G{\'o}recka, J. N. and G{\'o}recki, J.},
  title   = {T-shaped coincidence detector as a band filter of chemical signal frequency},
  journal = {Phys Rev E},
  year    = {2003},
  volume  = {67},
  number  = {6},
  pages   = {067203},
  doi     = {10.1103/PhysRevE.67.067203}
}

@article{gorecka2009simplest,
  author  = {G{\'o}recka, J. N. and G{\'o}recki, J. and Igarashi, Y.},
  title   = {On the simplest chemical signal diodes constructed with an excitable medium},
  journal = {Int J Unconv Comput},
  year    = {2009},
  volume  = {5},
  number  = {2},
  pages   = {129--143}
}

@article{gorecki2009information,
  author  = {G{\'o}recki, J. and G{\'o}recka, J. N. and Igarashi, Y.},
  title   = {Information processing with structured excitable medium},
  journal = {Nat Comput},
  year    = {2009},
  volume  = {8},
  number  = {3},
  pages   = {473--492},
  doi     = {10.1007/s11047-007-9032-5}
}

@article{homma2017fabrication,
  author  = {Homma, K. and Masuda, T. and Akimoto, A. M. and Nagase, K. and Itoga, K. and Okano, T. and Yoshida, R.},
  title   = {Fabrication of micropatterned self-oscillating polymer brush for direction control of chemical waves},
  journal = {Small},
  year    = {2017},
  volume  = {13},
  number  = {21},
  pages   = {1700041},
  doi     = {10.1002/smll.201700041}
}

@article{ito1992theory,
  author  = {Ito, H. and Glass, L.},
  title   = {Theory of reentrant excitation in a ring of cardiac tissue},
  journal = {Physica D},
  year    = {1992},
  volume  = {56},
  number  = {1},
  pages   = {84--106},
  doi     = {10.1016/0167-2789(92)90051-Y}
}

@article{keener1991eikonal,
  author  = {Keener, J. P.},
  title   = {An eikonal-curvature equation for action potential propagation in myocardium},
  journal = {J Math Biol},
  year    = {1991},
  volume  = {29},
  number  = {7},
  pages   = {629--651},
  doi     = {10.1007/BF00163916}
}

@article{keener2000homogenization,
  author  = {Keener, J. P.},
  title   = {Homogenization and propagation in the bistable equation},
  journal = {Physica D},
  year    = {2000},
  volume  = {136},
  number  = {1--2},
  pages   = {1--17},
  doi     = {10.1016/S0167-2789(99)00151-7}
}

@article{keener1992dynamics,
  author  = {Keener, J. P. and Tyson, J. J.},
  title   = {The dynamics of scroll waves in excitable media},
  journal = {SIAM Rev},
  year    = {1992},
  volume  = {34},
  number  = {1},
  pages   = {1--39},
  doi     = {10.1137/1034001}
}

@article{motoike2001realtime,
  author  = {Motoike, I. N. and Yoshikawa, K. and Iguchi, Y. and Nakata, S.},
  title   = {Real-time memory on an excitable field},
  journal = {Phys Rev E},
  year    = {2001},
  volume  = {63},
  number  = {3},
  pages   = {036220},
  doi     = {10.1103/PhysRevE.63.036220}
}

@article{sakurai2002design,
  author  = {Sakurai, T. and Mihaliuk, E. and Chirila, F. and Showalter, K.},
  title   = {Design and control of wave propagation patterns in excitable media},
  journal = {Science},
  year    = {2002},
  volume  = {296},
  number  = {5575},
  pages   = {2009--2012},
  doi     = {10.1126/science.1071265}
}

@article{sendinanadal2001wave,
  author  = {Sendi{\~n}a-Nadal, I. and Mihaliuk, E. and Wang, J. and P{\'e}rez-Mu{\~n}uzuri, V. and Showalter, K.},
  title   = {Wave propagation in subexcitable media with periodically modulated excitability},
  journal = {Phys Rev Lett},
  year    = {2001},
  volume  = {86},
  number  = {8},
  pages   = {1646--1649},
  doi     = {10.1103/PhysRevLett.86.1646}
}

@article{sendinanadal2006kinematic,
  author  = {Sendi{\~n}a-Nadal, I. and deCastro, M. and G{\'o}mez-Gesteira, M.},
  title   = {Kinematic description of wave propagation through a chemical diode},
  journal = {Chaos},
  year    = {2006},
  volume  = {16},
  number  = {3},
  pages   = {033110},
  doi     = {10.1063/1.2221530}
}

@article{steinbock1995navigating,
  author  = {Steinbock, O. and T{\'o}th, {\'A}. and Showalter, K.},
  title   = {Navigating complex labyrinths: optimal paths from chemical waves},
  journal = {Science},
  year    = {1995},
  volume  = {267},
  number  = {5199},
  pages   = {868--871},
  doi     = {10.1126/science.267.5199.868}
}

@article{steinbock1995anisotropy,
  author  = {Steinbock, O. and Kettunen, P. and Showalter, K.},
  title   = {Anisotropy and spiral organizing centers in patterned excitable media},
  journal = {Science},
  year    = {1995},
  volume  = {269},
  number  = {5232},
  pages   = {1857--1860},
  doi     = {10.1126/science.269.5232.1857}
}

@article{toth1995logic,
  author  = {T{\'o}th, {\'A}. and Showalter, K.},
  title   = {Logic gates in excitable media},
  journal = {J Chem Phys},
  year    = {1995},
  volume  = {103},
  number  = {6},
  pages   = {2058--2066},
  doi     = {10.1063/1.469732}
}

@article{tyson1988singular,
  author  = {Tyson, J. J. and Keener, J. P.},
  title   = {Singular perturbation theory of traveling waves in excitable media (a review)},
  journal = {Physica D},
  year    = {1988},
  volume  = {32},
  number  = {3},
  pages   = {327--361},
  doi     = {10.1016/0167-2789(88)90062-0}
}

@article{xin2000front,
  author  = {Xin, J.},
  title   = {Front propagation in heterogeneous media},
  journal = {SIAM Rev},
  year    = {2000},
  volume  = {42},
  number  = {2},
  pages   = {161--230},
  doi     = {10.1137/S0036144500364296}
}
```

**DOI caveat:** the following DOIs are taken from secondary reference lists, not from a
Crossref lookup in this session, and should be sanity-checked when first cited:
`collifranzone1990wavefront`, `gorecki2009information`, `ito1992theory`,
`keener1992dynamics`, `sakurai2002design`, `steinbock1995navigating`, `xin2000front`.
Everything else (authors/title/venue/volume/pages/year) was verified against Crossref,
PubMed, journal tables of contents, or the author's own publications page.
