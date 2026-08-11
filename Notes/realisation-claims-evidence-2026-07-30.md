# Realisation claims for BZ computing: literature evidence (2026-07-30)

Purpose: verify the practical-realisation claims made in project discussions about
BZ/excitable-media computing against the published literature. Every entry below was
checked by web search and/or by fetching the publisher/abstract page; title, venue,
year and page/article numbers all check out. DOIs are given where they could be
verified or are the registered publisher DOI; a few entries deliberately carry no DOI
because it could not be verified (do not invent them).

## Claim 1 — BZ wave-computing experiments run in immobilised media without bulk flow, so pure reaction-diffusion models apply

Verdict: SUPPORTED (with one honest caveat about buoyancy convection in gel-free layers).

- Steinbock, Toth, Showalter (1995), Science 267, 868-871. The maze-solving
  experiment used the ferroin-catalysed BZ reaction immobilised in a polyacrylamide
  gel layer; waves navigate a printed maze purely by reaction-diffusion. No
  through-flow exists anywhere in the setup.
- Adamatzky & De Lacy Costello (2002), Phys. Rev. E 66, 046112. The XOR gate was
  demonstrated in a thin, unstirred layer of BZ reagent with wave fragments
  injected by silver-wire triggers; computation is by collision/annihilation in a
  quiescent film.
- Suematsu & Nakata (2021), Materials 14, 6177. Photosensitive BZ on a BZ-soaked
  filter paper covered with silicone oil; again an immobilised, flow-free medium.
  They model it with a photosensitive Oregonator that contains only reaction and
  diffusion terms — direct precedent for our modelling choice.
- Yamaguchi, Kuhnert, Nagy-Ungvarai, Muller, Hess (1991), J. Phys. Chem. 95,
  5831-5837. The standard methods paper for running BZ in gel matrices
  (silica/polyacrylamide), establishing the immobilised-medium protocol used by
  most subsequent wave-computing experiments.

Caveat: Cliffe, Tavener & Wilke (1998), Phys. Fluids 10, 730 show that in gel-free
vertical channels, buoyancy-driven convection from concentration gradients can
spontaneously arise and change the wave speed — so "no advection" is a property of
the immobilised/thin-film setups, not of BZ chemistry in general.

For the deck: "Published BZ wave computers run in gels, membranes or BZ-soaked
paper — quiescent media where a pure reaction-diffusion model is the standard and
appropriate description (Steinbock 1995; Adamatzky 2002; Suematsu 2021)."

## Claim 2 — Photosensitive BZ allows spatial light patterns as a programmable input/weight field, including projector-based illumination

Verdict: SUPPORTED.

- Kuhnert (1986), Nature 319, 393-394. First demonstration that illumination of
  the Ru(bpy)3-catalysed BZ medium writes/erases wave patterns — an "optical
  photochemical memory device"; the origin of light as a programmable field.
- Kuhnert, Agladze, Krinsky (1989), Nature 337, 244-247. Patterned illumination of
  the light-sensitive BZ medium performs image processing (contrast enhancement,
  contour extraction), showing a spatial light pattern acts as a functional input.
- Sakurai, Mihaliuk, Chirila, Showalter (2002), Science 296, 2009-2012.
  Computer-controlled spatiotemporal illumination used as feedback to design and
  steer wave propagation paths in photosensitive BZ — the most direct precedent
  for a phi(x,y,t) control field.
- Suematsu & Nakata (2021), Materials 14, 6177. Explicitly uses a commercial
  Epson liquid-crystal projector driven from PowerPoint grayscale images to impose
  arbitrary light fields (photoexcitation AND photoinhibition under the same
  chemistry) on the BZ medium — exactly the "projector as programmer" claim.

For the deck: "In photosensitive (Ru-catalysed) BZ, a projector-drawn light field
is a proven, rewritable control/input field — from Kuhnert 1986 to modern
PowerPoint-driven projector experiments (Suematsu 2021)."

## Claim 3 — Readout is optical and measurable with simple photodiodes/cameras without digital computation

Verdict: SUPPORTED, with a caveat: published gate papers usually digitise the
images for analysis, even though the signal itself is directly visible/thresholdable.

- Wood & Ross (1985), J. Chem. Phys. 82, 1924-1936. BZ wave profiles recorded
  quantitatively with a linear photodiode array reading ferroin light absorption —
  proof that the wave state is a clean analogue optical signal readable with
  minimal electronics (no computer required for detection).
- Adamatzky & De Lacy Costello (2002) and Steinbock et al. (1995): readout is the
  ferroin colour change (red/blue), recorded by video camera; the presence of a
  wave at an output channel is directly visible to the naked eye.
- Duenas-Diez & Perez-Mercader (2021), Front. Chem. 9, 611120. Their BZ Turing
  machine is read out from the raw redox-potential (electrode) trace, a
  single-channel electrical analogue of the optical state — an alternative
  non-computational readout.
- Counterpoint: Sharma et al. (2024), Nat. Commun. 15, 1984 found it necessary to
  couple BZ oscillator arrays to digital electronics for error correction —
  optical readout is easy, but reliable multi-stage computation pushed them to a
  hybrid scheme.

For the deck: "The BZ state is self-indicating: ferroin colour or Ru fluorescence
is readable by eye, photodiode or redox electrode (Wood & Ross 1985) — no digital
hardware is needed for the measurement itself."

## Claim 4 — Wave speeds ~0.1-1 mm/s and batch reagent lifetimes of hours

Verdict: PARTIALLY SUPPORTED. The 0.1 mm/s end is right; 1 mm/s is an overestimate
(roughly 5x too fast at room temperature). Batch lifetimes of tens of minutes to
hours are consistent with the literature, though not all media survive that long.

- Suematsu & Nakata (2021), Materials 14, 6177. Directly measured wave speeds of
  approx. 0.15-0.20 mm/s in photosensitive BZ on filter paper. Also states the
  experiment was limited to about 5 min observation "owing to the batch system" —
  i.e., batch media are time-limited.
- Wood & Ross (1985), J. Chem. Phys. 82, 1924-1936. Systematic velocity
  measurements in quiescent BZ layers; speeds of a few mm/min (order 0.05-0.15
  mm/s) depending on recipe and temperature — consistent with the low end of the
  claimed range.
- General context: the BZ oscillation in a closed batch system persists through
  many cycles (Scholarpedia/Epstein reviews cite up to thousands of cycles in
  well-mixed batch reactors), so "hours" is fair for well-mixed batch chemistry,
  but thin-film/gel computing substrates typically give tens of minutes to ~1-2 h
  of usable excitable behaviour.

For the deck: "BZ waves crawl at order 0.1 mm/s (0.15-0.2 mm/s measured, Suematsu
2021) and a batch substrate stays usable for tens of minutes — chemistry is slow,
so the selling point must be parallelism, not speed."

## Claim 5 — In flow reactors/droplets, advection materially changes wave dynamics (flow-fed realisations need advective terms)

Verdict: SUPPORTED.

- Paoletti & Solomon (2005), Phys. Rev. E 72, 046204. Excitable Ru-catalysed BZ
  fronts in a chain of vortices are mode-locked, pinned, stretched and even erased
  by the imposed flow — advection qualitatively reorganises the front dynamics.
- Cliffe, Tavener & Wilke (1998), Phys. Fluids 10, 730-731. Even without imposed
  pumping, buoyancy-driven convection in a narrow vertical channel arises at a
  symmetry-breaking bifurcation and alters the BZ wave speed — convection is not
  a small perturbation.
- Kitahata, Aihara, Magome, Yoshikawa (2002), J. Chem. Phys. 116, 5666-5672.
  A BZ wave itself drives Marangoni convection at a free surface, producing
  convective motion coupled to the wave — reaction-transport coupling runs both
  ways in flow-capable setups.
- Fullarton, Draper, Phillips, de Lacy Costello, Adamatzky (2019), J. Phys.:
  Mater. 2, 015005. BZ in liquid marbles and Torbensen et al. (2017), Lab Chip
  17 (BZ droplet emulsion networks made by microfluidics): once the chemistry is
  compartmentalised in mobile/carried droplets, hydrodynamic transport and
  inter-droplet coupling become part of the dynamics, unlike gel media.

For the deck: "Add flow and the wave dynamics change qualitatively — fronts are
dragged, pinned, mode-locked or erased (Paoletti & Solomon 2005); a flow-fed
realisation must include advection, a gel realisation need not."

## Claim 6 — Chemical/analog memory and readout tricks exist that avoid electronic readout

Verdict: SUPPORTED.

- Motoike, Yoshikawa, Iguchi, Nakata (2001), Phys. Rev. E 63, 036220. "Real-time
  memory on an excitable field": the refractory trace left by a wave in an
  excitable BZ field stores which path was taken, and a later probe wave reads it
  out — a latch built purely from excitability.
- Kaminaga, Vanag, Epstein (2006), Angew. Chem. Int. Ed. 45, 3087-3089. A
  reaction-diffusion memory device: illumination writes a Turing-pattern state in
  a microemulsion BZ-AOT system that persists after the writing light is removed.
- Gizynski & Gorecki (2017), Phys. Chem. Chem. Phys. 19, 6519-6531. Memory states
  coded in light-controlled oscillations of interacting BZ droplets; information
  is stored in the phase configuration and survives for many cycles.
- Gorecki, Yoshikawa, Igarashi (2003), J. Phys. Chem. A 107, 1664-1669. Chemical
  reactors that count: excitable-pulse trains are counted by downstream chemistry
  — an integrating/accumulator stage without electronics.

For the deck: "Excitable media have native memory: refractory trails (Motoike
2001), light-written Turing states (Kaminaga 2006), droplet phase memory
(Gizynski 2017) and pulse-counting reactors (Gorecki 2003) — readout does not
have to be electronic."

## Claim 7 — Counter-evidence: documented practical difficulties of BZ/excitable-media computing

Verdict: SUPPORTED — the difficulties are real and documented by the field's own
proponents; any honest deck must state them.

- Adamatzky (2019), Phil. Trans. R. Soc. B 374, 20180372. Historical review by
  the field's leading proponent; catalogues why liquid/RD computers have not
  scaled: slow speeds, one-shot substrates, reproducibility and interfacing
  problems.
- Duenas-Diez & Perez-Mercader (2021), Front. Chem. 9, 611120. Explicitly argues
  cascaded Boolean gates are "prohibitive" in chemistry because information is
  carried by mass transport (diffusion/convection), which irreversibly degrades
  it between gates — this is precisely the non-cascadability argument. Their
  workaround is automata-style (Turing machine) computation in a single pot.
- Sharma, Ng, Parrilla Gutierrez, Jiang, Cronin (2024), Nat. Commun. 15, 1984.
  State-of-the-art BZ processor; its very architecture is an admission of
  difficulty — probabilistic chemical logic plus digital error correction was
  required to get reliable computation from BZ oscillator arrays.
- Gorecki, Gizynski, Guzowski, Gorecka, Garstecki, Gruenert, Dittrich (2015),
  Phil. Trans. R. Soc. A 373, 20140219. Review of chemical computing with RD
  processes; discusses droplet-based architectures partly as a response to the
  drift, reagent-consumption and one-shot limitations of extended BZ media.

For the deck: "The honest limitations: waves are ~10^6x slower than electronics,
substrates are consumed, gates do not cascade cleanly (Duenas-Diez 2021), and the
best modern BZ computer needed digital error correction to work (Sharma 2024)."

## BibTeX (verified entries)

@article{steinbock1995navigating,
  author  = {Steinbock, Oliver and T{\'o}th, {\'A}gota and Showalter, Kenneth},
  title   = {Navigating complex labyrinths: optimal paths from chemical waves},
  journal = {Science},
  volume  = {267}, number = {5199}, pages = {868--871}, year = {1995},
  doi     = {10.1126/science.267.5199.868}}

@article{toth1995logic,
  author  = {T{\'o}th, {\'A}gota and Showalter, Kenneth},
  title   = {Logic gates in excitable media},
  journal = {J. Chem. Phys.},
  volume  = {103}, number = {6}, pages = {2058--2066}, year = {1995},
  doi     = {10.1063/1.470010}}

@article{adamatzky2002experimental,
  author  = {Adamatzky, Andrew and De Lacy Costello, Benjamin P. J.},
  title   = {Experimental logical gates in a reaction-diffusion medium: The {XOR} gate and beyond},
  journal = {Phys. Rev. E},
  volume  = {66}, pages = {046112}, year = {2002},
  doi     = {10.1103/PhysRevE.66.046112}}

@article{suematsu2021instability,
  author  = {Suematsu, Nobuhiko J. and Nakata, Satoshi},
  title   = {Instability of the homogeneous distribution of chemical waves in the {Belousov-Zhabotinsky} reaction},
  journal = {Materials},
  volume  = {14}, number = {20}, pages = {6177}, year = {2021},
  doi     = {10.3390/ma14206177}}

@article{yamaguchi1991gel,
  author  = {Yamaguchi, Tomohiko and Kuhnert, Lothar and Nagy-Ungvarai, Zsuzsanna and M{\"u}ller, Stefan C. and Hess, Benno},
  title   = {Gel systems for the {Belousov-Zhabotinskii} reaction},
  journal = {J. Phys. Chem.},
  volume  = {95}, number = {15}, pages = {5831--5837}, year = {1991}}

@article{kuhnert1986new,
  author  = {Kuhnert, Lothar},
  title   = {A new optical photochemical memory device in a light-sensitive chemical active medium},
  journal = {Nature},
  volume  = {319}, pages = {393--394}, year = {1986},
  doi     = {10.1038/319393a0}}

@article{kuhnert1989image,
  author  = {Kuhnert, Lothar and Agladze, Konstantin I. and Krinsky, Valentin I.},
  title   = {Image processing using light-sensitive chemical waves},
  journal = {Nature},
  volume  = {337}, pages = {244--247}, year = {1989},
  doi     = {10.1038/337244a0}}

@article{sakurai2002design,
  author  = {Sakurai, Tatsunari and Mihaliuk, Eduard and Chirila, Florentin and Showalter, Kenneth},
  title   = {Design and control of wave propagation patterns in excitable media},
  journal = {Science},
  volume  = {296}, number = {5575}, pages = {2009--2012}, year = {2002},
  doi     = {10.1126/science.1071265}}

@article{woodross1985quantitative,
  author  = {Wood, Paul M. and Ross, John},
  title   = {A quantitative study of chemical waves in the {Belousov-Zhabotinsky} reaction},
  journal = {J. Chem. Phys.},
  volume  = {82}, number = {4}, pages = {1924--1936}, year = {1985}}

@article{duenasdiez2021native,
  author  = {Due{\~n}as-D{\'i}ez, Marta and P{\'e}rez-Mercader, Juan},
  title   = {Native chemical computation. {A} generic application of oscillating chemistry illustrated with the {Belousov-Zhabotinsky} reaction. {A} review},
  journal = {Front. Chem.},
  volume  = {9}, pages = {611120}, year = {2021},
  doi     = {10.3389/fchem.2021.611120}}

@article{sharma2024programmable,
  author  = {Sharma, Abhishek and Ng, Marcus Tze-Kiat and Parrilla Gutierrez, Juan Manuel and Jiang, Yibin and Cronin, Leroy},
  title   = {A programmable hybrid digital chemical information processor based on the {Belousov-Zhabotinsky} reaction},
  journal = {Nat. Commun.},
  volume  = {15}, pages = {1984}, year = {2024},
  doi     = {10.1038/s41467-024-45896-7}}

@article{paoletti2005front,
  author  = {Paoletti, Matthew S. and Solomon, Thomas H.},
  title   = {Front propagation and mode-locking in an advection-reaction-diffusion system},
  journal = {Phys. Rev. E},
  volume  = {72}, pages = {046204}, year = {2005},
  doi     = {10.1103/PhysRevE.72.046204}}

@article{cliffe1998convective,
  author  = {Cliffe, K. A. and Tavener, S. J. and Wilke, H.},
  title   = {Convective effects on a propagating reaction front},
  journal = {Phys. Fluids},
  volume  = {10}, number = {3}, pages = {730--731}, year = {1998}}

@article{kitahata2002convective,
  author  = {Kitahata, Hiroyuki and Aihara, Ryo and Magome, Nobuyuki and Yoshikawa, Kenichi},
  title   = {Convective and periodic motion driven by a chemical wave},
  journal = {J. Chem. Phys.},
  volume  = {116}, number = {13}, pages = {5666--5672}, year = {2002}}

@article{fullarton2019belousov,
  author  = {Fullarton, Claire and Draper, Thomas C. and Phillips, Neil and de Lacy Costello, Ben P. J. and Adamatzky, Andrew},
  title   = {Belousov-{Zhabotinsky} reaction in liquid marbles},
  journal = {J. Phys.: Mater.},
  volume  = {2}, number = {1}, pages = {015005}, year = {2019}}

@article{torbensen2017chemical,
  author  = {Torbensen, Kristian and Rossi, Federico and Ristori, Sandra and Abou-Hassan, Ali},
  title   = {Chemical communication and dynamics of droplet emulsions in networks of {Belousov-Zhabotinsky} micro-oscillators produced by microfluidics},
  journal = {Lab Chip},
  volume  = {17}, pages = {1179--1189}, year = {2017},
  doi     = {10.1039/C6LC01583B}}

@article{motoike2001realtime,
  author  = {Motoike, Ikuko N. and Yoshikawa, Kenichi and Iguchi, Yasuhiro and Nakata, Satoshi},
  title   = {Real-time memory on an excitable field},
  journal = {Phys. Rev. E},
  volume  = {63}, pages = {036220}, year = {2001},
  doi     = {10.1103/PhysRevE.63.036220}}

@article{kaminaga2006reaction,
  author  = {Kaminaga, Akiko and Vanag, Vladimir K. and Epstein, Irving R.},
  title   = {A reaction-diffusion memory device},
  journal = {Angew. Chem. Int. Ed.},
  volume  = {45}, number = {19}, pages = {3087--3089}, year = {2006}}

@article{gizynski2017chemical,
  author  = {Gizynski, Konrad and Gorecki, Jerzy},
  title   = {Chemical memory with states coded in light controlled oscillations of interacting {Belousov-Zhabotinsky} droplets},
  journal = {Phys. Chem. Chem. Phys.},
  volume  = {19}, pages = {6519--6531}, year = {2017}}

@article{gorecki2003chemical,
  author  = {Gorecki, Jerzy and Yoshikawa, Kenichi and Igarashi, Yasuhiro},
  title   = {On chemical reactors that can count},
  journal = {J. Phys. Chem. A},
  volume  = {107}, number = {10}, pages = {1664--1669}, year = {2003}}

@article{adamatzky2019brief,
  author  = {Adamatzky, Andrew},
  title   = {A brief history of liquid computers},
  journal = {Phil. Trans. R. Soc. B},
  volume  = {374}, number = {1774}, pages = {20180372}, year = {2019},
  doi     = {10.1098/rstb.2018.0372}}

@article{gorecki2015chemical,
  author  = {Gorecki, Jerzy and Gizynski, Konrad and Guzowski, Jan and Gorecka, Joanna N. and Garstecki, Piotr and Gruenert, Gerd and Dittrich, Peter},
  title   = {Chemical computing with reaction-diffusion processes},
  journal = {Phil. Trans. R. Soc. A},
  volume  = {373}, number = {2046}, pages = {20140219}, year = {2015},
  doi     = {10.1098/rsta.2014.0219}}

## Addendum (2026-07-30): Tyson-Fife scaling reference values

Source: Scholarpedia "Oregonator" article (drawing on Tyson & Fife, J. Chem. Phys. 73, 2224 (1980) and Tyson, in "Oscillations and Traveling Waves in Chemical Systems", Wiley 1985), with FKN rate constants from Field & Forsterling (1986) at [H+] = 0.8 M.

Underlying rate constants (FKN, acidity-adjusted): k1 = 1.28, k2 = 2.4e6, k3 = 33.6, k4 = 2400 (all M^-1 s^-1); kc = 1 M^-1 s^-1. Reference composition: A = [BrO3-] = 0.06 M, B = [malonic acid] = 0.02 M, [H+] = 0.8 M.

Scaling definitions:
  x   = 2 k4 X / (k3 A)              (activator, X = HBrO2)
  y   = k2 Y / (k3 A)                (bromide)
  z   = kc k4 B Z / (k3 A)^2         (catalyst, Z = Ce(IV))
  tau = kc B t                       (dimensionless time)

Canonical dimensionless parameters (FKN-fitted):
  eps  = kc B / (k3 A)          = 9.90e-3
  eps' = 2 kc k4 B / (k2 k3 A)  = 1.98e-5
  q    = 2 k1 k4 / (k2 k3)      = 7.62e-5
  f    = stoichiometric factor (0..3; ~1 in canonical runs)

IMPORTANT CONVENTION NOTE: our simulations use the *computational* parameter set
common in excitable-media theory (eps = 0.05, q = 0.002, f = 1.4), NOT the
FKN-fitted canonical values above (eps = 9.9e-3, q = 7.6e-5). The computational
set preserves the dynamics while giving convenient numerics; it is the standard
choice in the wave-simulation literature. Any SI mapping of model units should
therefore be treated as calibration, not as an FKN identity.

Time scale with canonical values: tau = kc B t = 0.02 t, so 1 t.u. ~ 50 s
(recipe-dependent: kc, B, and acidity all move this). Length scale:
L0 = sqrt(D * T0); with D ~ 2e-5 cm^2/s and T0 ~ 50 s, L0 ~ 0.3 mm. Cross-check:
model c ~ 6.2 L0/t.u. maps to ~0.04 mm/s, within a factor ~4 of the measured
photosensitive-BZ speed (0.15-0.2 mm/s, Suematsu 2021) given recipe dependence.
