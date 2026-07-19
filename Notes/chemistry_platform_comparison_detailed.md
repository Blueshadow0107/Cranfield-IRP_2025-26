# Detailed comparison of programmable chemical computing platforms

Date: 2026-06-18
Purpose: Expand on alternative chemistries for trainable reaction networks and reaction-diffusion computation.

---

## 1. DNA strand displacement (DSD)

### What it is
Short single-stranded DNA overhangs (toeholds) initiate the displacement of a bound strand by an invading strand. The reaction is enzyme-free and operates at room temperature in aqueous buffer.

### Key elementary step
```
Input + Gate  <->  Input:Gate*  ->  Output + Waste
       (toehold bind)  (branch migrate)
```

### How you control it
| Knob | Effect | Range |
|------|--------|-------|
| Toehold length | Binding energy | ~1 nt ≈ 10x rate change |
| Toehold sequence (GC vs AT) | Binding strength | ~10²–10³ fold |
| Mismatches / remote toeholds | Kinetic barriers | ~10³–10⁵ fold |
| Strand concentrations | Effective weight | Continuous, 1 nM – 10 µM |

### Timescales
- Bimolecular rate limit: ~10⁶–10⁷ M⁻¹s⁻¹
- Effective reaction times: seconds to hours
- Typical concentrations: 1 nM – 1 µM

### Pros
- Enzyme-free; only DNA strands needed
- Enormous independent rate tunability
- Well-developed design tools (NUPACK, Visual DSD)
- Experimentally demonstrated for logic, neural networks, pattern recognition

### Cons
- DNA is expensive at scale
- Leak reactions and crosstalk limit performance
- Sequence design is non-trivial
- Slower than enzymatic or electronic systems

### Key examples
- Qian & Winfree 2011: seesaw neural network
- Cherry & Qian 2018: winner-take-all DNA network classifying MNIST digits
- Lee, Buse & Winfree 2025: differentiable programming of indexed CRNs

---

## 2. Enzymatic DNA toolbox (PEN toolbox)

### What it is
A set of DNA templates plus three enzymes: a DNA polymerase (P), a nickase (N), and an exonuclease (E). The enzymes copy, cut, and degrade DNA strands, creating out-of-equilibrium dynamics.

### Key reactions
```
S + T   --polymerase-->   extended S:T
extended S:T --nickase-->  S + T + product
S       --exonuclease-->   ∅
```

### How you control it
| Knob | Effect | Range |
|------|--------|-------|
| Template sequence | Reaction specificity | Binary (matches / no match) |
| Template concentration | Effective rate / weight | Continuous |
| Enzyme concentration | Overall speed and saturation | Continuous |
| Temperature | Enzyme activity | Typically 37–55 °C |

### Timescales
- Polymerase extension: ~10–1000 nt/s
- Nickase cutting: fast
- Exonuclease degradation: sets lifetime, minutes to hours
- Network dynamics: minutes to hours

### Pros
- Faster and more robust than enzyme-free DNA
- Autocatalysis and bistability are easy to implement
- Works in microfluidic droplets
- Template concentrations directly map to weights

### Cons
- Requires purified enzymes and careful buffer conditions
- Enzyme saturation and degradation limit dynamic range
- Protein handling complicates experiments

### Key examples
- Rondelez group: PEN toolbox for oscillators, bistable switches, pattern formation
- Okumura et al. 2022: enzymatic neural networks in microdroplets
- Zambrano et al. 2022: PEN reactions in synthetic cells (proteinosomes)

---

## 3. Small-molecule reaction-diffusion systems

### What it is
Ordinary chemical oscillators and pattern-forming systems, most famously the Belousov-Zhabotinsky (BZ) reaction, where an organic substrate (malonic acid) is oxidised by bromate in the presence of a metal catalyst.

### Key reactions (Oregonator model, simplified)
```
BrO3- + Br- + H+  ->  HBrO2 + HOBr      (inhibition)
HBrO2 + BrO3- + H+  ->  2 HBrO2 + ...   (autocatalysis)
HBrO2  ->  Br-                            (radical termination)
```

### How you control it
| Knob | Effect | Range |
|------|--------|-------|
| Temperature | Overall reaction rate | ~2x per 10 °C |
| Catalyst (Ce, Ru, ferroin) | Wavelength / sensitivity | Discrete choices |
| Electric field | Wave speed and direction | Global |
| Geometry / gel structure | Wave propagation | Pattern-level |
| Light (for photosensitive variants) | Local inhibition | Global or patterned |

### Timescales
- Oscillation period: seconds to minutes
- Wave propagation: mm/min
- Pattern formation: minutes to hours

### Pros
- Very cheap and simple reagents
- Beautiful, well-studied spatial dynamics
- No sequence design needed
- Excellent for reaction-diffusion demonstrations

### Cons
- Rates are set by intrinsic chemistry; you cannot independently tune individual "weights"
- Difficult to implement arbitrary logic or trainable networks
- Reactions are complex and can be temperamental

### Key examples
- Adamatzky & Costello: reaction-diffusion computing with BZ
- Parrilla-Gutierrez et al. 2020: programmable BZ chemical computer with memory
- Sharma et al. 2024: hybrid digital chemical processor based on BZ

---

## 4. Photochemical / photoswitchable reaction networks

### What it is
Molecules that change shape or reactivity when illuminated, allowing light to switch reaction pathways on and off. Common switches include azobenzenes, diarylethenes, and spiropyrans.

### Key idea
```
A_trans + B  --hv-->  A_cis + B
A_cis has different binding / catalytic activity than A_trans
```

### How you control it
| Knob | Effect | Range |
|------|--------|-------|
| Light wavelength | Select which isomer | Discrete bands |
| Light intensity | Switching rate | Continuous |
| Illumination pattern | Spatial programmability | Arbitrary 2D/3D |
| Dark relaxation | Natural timescale | ms to hours |

### Timescimescales
- Photoisomerisation: ps to ms
- Subsequent reaction: depends on chemistry
- Network dynamics: ms to hours

### Pros
- Non-invasive, wireless control
- Excellent spatial and temporal resolution
- Can reconfigure a network in real time

### Cons
- Photoswitches add synthetic complexity
- Light penetration limits thick samples
- Typically used as a control layer, not as the main compute medium

### Key examples
- Merckx et al. 2024: phosphate fuels + azobenzene photoswitches for reaction networks
- Liquid crystal polymer networks: light-driven mechanical computation

---

## 5. Transcriptional / cell-free gene circuits

### What it is
DNA templates are transcribed into RNA by RNA polymerase, and RNA is translated into proteins by ribosomes. The proteins then regulate further transcription. Cell-free extracts allow this outside living cells.

### Key reactions
```
DNA_i --RNAP--> mRNA_i
mRNA_i --ribosome--> Protein_i
Protein_i + DNA_j --regulates--> transcription of DNA_j
```

### How you control it
| Knob | Effect | Range |
|------|--------|-------|
| Promoter sequence | Transcription rate | 10²–10⁴ fold |
| Ribosome binding site | Translation rate | 10²–10³ fold |
| Transcription factor design | Regulatory logic | Modular |
| Degradation tags | Protein lifetime | 10² fold |

### Timescales
- Transcription: minutes
- Translation: minutes to tens of minutes
- Protein lifetime: tens of minutes to hours
- Network dynamics: hours to days

### Pros
- Highly modular and programmable
- Natural building blocks for feedback and regulation
- Can implement complex logic

### Cons
- Very slow compared to small-molecule or DNA systems
- Cell-free extracts are complex and variable
- Hard to image and control spatially

### Key examples
- Kim & Winfree 2006: in vitro bistable circuit from transcriptional switches
- Franco et al. 2011: synthetic transcriptional clock
- Synthetic gene regulatory networks in general

---

## 6. Electrochemical / memristive systems

### What it is
Electrode reactions or redox-active materials whose conductance or state depends on the history of applied voltage. Memristors are the electronic analogue: resistance depends on charge that has passed through.

### Key idea
```
M_ox + e-  <->  M_red
Applied potential sets oxidation state, which changes conductivity
```

### How you control it
| Knob | Effect | Range |
|------|--------|-------|
| Electrode potential | Redox state | Continuous |
| Voltage pulse history | Memristor weight | Continuous, non-volatile |
| Electrode geometry | Spatial resolution | Lithography-limited |

### Timescales
- Electron transfer: ns to µs
- Ion migration in memristors: ns to ms
- Network dynamics: can be very fast

### Pros
- Extremely fast
- Direct electrical readout
- Mature fabrication techniques
- Natural analogy to neural-network weights

### Cons
- Not a "chemical reaction network" in the traditional sense
- Hard to make spatially continuous analog fields
- Different physics than molecular CRNs

### Key examples
- Metal-oxide memristors (HP Labs, Strukov et al.)
- Electrochemical neuromorphic devices
- Redox-based reservoir computing

---

## 7. Colloidal and soft-matter systems

### What it is
Particles, droplets, or active matter that interact through chemical gradients, binding, or mechanical forces. Computation emerges from collective dynamics rather than designed reactions.

### Key idea
```
Particles release / consume chemicals -> diffusion gradients -> attraction / repulsion
```

### How you control it
| Knob | Effect | Range |
|------|--------|-------|
| Particle size / charge | Interaction strength | Continuous |
| Surface chemistry | Binding specificity | Discrete choices |
| External fields (light, magnetic) | Driving force | Global or patterned |
| Geometry | Boundary conditions | Macroscopic design |

### Timescales
- Diffusion-limited interactions: ms to hours
- Collective dynamics: seconds to hours

### Pros
- Can self-organise into patterns and structures
- Direct visual readout
- Scales to macroscopic systems

### Cons
- Difficult to program precise weights or logic
- Computation is emergent, not designed
- Hard to train

### Key examples
- Active matter droplets (Pascal/Magnasco style)
- DNA-coated colloids
- Self-propelled particle systems

---

## Summary: which platform for which goal?

| Goal | Recommended platform | Why |
|------|---------------------|-----|
| Trainable CRN with arbitrary weights | DNA strand displacement or enzymatic PEN | Independent rate tuning by sequence or concentration |
| Fast, robust classification in droplets | Enzymatic PEN | Bistability + droplet compatibility |
| Reaction-diffusion pattern formation | BZ or small-molecule RD | Cheap, well-understood spatial dynamics |
| Reconfigurable / real-time control | Photochemical switches | Light as a global or spatial control knob |
| Complex biological-style regulation | Transcriptional circuits | Natural modularity, but slow |
| Fast analogue weights | Electrochemical memristors | Speed and non-volatile weights |
| Emergent / self-organised computation | Colloidal / active matter | Scales visually, but less programmable |
