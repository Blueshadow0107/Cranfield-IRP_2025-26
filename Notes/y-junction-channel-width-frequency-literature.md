# Y-Junction Channel Width vs. Signal Frequency — Literature Summary

*Date: 2026-05-08*  
*Context: Cranfield IRP 2025-26 — Acoustic Y-junction logic unit (FDTD)*

---

## Key Reference (Directly Relevant)

**Wang et al. (2019), "Binary-phase acoustic passive logic gates", *Scientific Reports* 9, 8355.**  
DOI: [10.1038/s41598-019-44769-0](https://doi.org/10.1038/s41598-019-44769-0)

- **Authors:** Yin Wang, Jian-ping Xia, Hong-xiang Sun, Shou-qi Yuan, Xiao-jun Liu
- **Geometry:** Tri-port waveguide (two inputs A & B, one output O) — essentially a Y-junction.
- **Mechanism:** Linear acoustic interference with binary-phase passive unit cells (Helmholtz resonator arrays) placed in the waveguide to manipulate phase.
- **Material:** Epoxy resin 3D-printed structure; air-filled channels ($\rho_{\text{air}} = 1.21$ kg/m³, $c_{\text{air}} = 343$ m/s).
- **Logic demonstrated:** OR, NOT, AND, XOR (basic); XNOR, NOR, NAND, A$\odot$(B+C) (composite).
- **Operating frequency:** **3.43 kHz** ($\lambda = 10$ cm in air).
- **Gate dimensions:** **0.6$\lambda$ length $\times$ 0.3$\lambda$ width** (i.e. 60 mm $\times$ 30 mm).
- **Unit cell:** $l = \lambda/2$, $w = \lambda/10$.
- **Bandwidth:** **~0.2 fractional bandwidth** (0.17 for AND gate).
- **Logic threshold:** Uniform at **0.4 Pa** output pressure amplitude (input amplitude 1.0 Pa).
- **Critical insight:** The logic is implemented purely by **phase manipulation** of incoming waves — exactly the physics your FDTD step 4 aims to capture.

> **Note:** The 0.3$\lambda$ width and 0.6$\lambda$ length are **dimensionless design rules** that scale to any medium and frequency. Our FDTD simulation applies these ratios to water at 1 MHz, yielding a 0.45 mm channel width and 0.9 mm gate length.

---

## General Acoustic Waveguide Design Rules

### 1. Modal Cutoff

| Waveguide shape | Dominant mode cutoff | Rule |
|-----------------|---------------------|------|
| Cylindrical | $\lambda_c \approx 1.71 D$ | First higher mode cuts off when $\lambda > 1.71\times$ diameter |
| Rectangular | $\lambda_c = 2a$ | TE10 cuts off when $\lambda > 2\times$ wide dimension |
| General | — | Keep width < ~0.5$\lambda$ to suppress higher modes |

- The **plane-wave (0,0) mode** has **no cutoff** — it propagates at all frequencies with linear dispersion $\omega = kc$.
- For a clean single-mode logic gate, you want only the plane wave. Wang et al. achieve this at **0.3$\lambda$ width**.

### 2. Microfluidic / Acoustic Channel Design Rules

| Source | Rule | Context |
|--------|------|---------|
| BAW device literature [36,52–54] | $w_{\text{ch}} = \lambda_{\text{med}} / 2$ | Half-wavelength resonant channel |
| SAW microfluidics (Friend & Yeo) | $w_{\text{ch}} \geq 7$–$10\,\lambda$ | Avoid diffraction during SAW propagation |
| Wang et al. (2019) | $w_{\text{ch}} \approx 0.3\,\lambda$ | Sub-wavelength compact logic gate |

**Tension:** SAW devices need wide channels (many $\lambda$) to host multiple pressure nodes. Bulk-wave logic gates can be **sub-wavelength** (0.3$\lambda$) because they rely on interference in a compact region, not on establishing a standing wave across the channel.

### 3. Attenuation Considerations

- Viscous and thermal boundary-layer damping: $\alpha \propto \omega^{1/2} / a$
- Narrower channels $\rightarrow$ higher damping per unit length.
- For water at room temperature, this is usually negligible for mm-scale channels below ~1 MHz, but becomes relevant for sub-mm channels or frequencies above a few MHz.

---

## Application to Your FDTD Parameters

**Our simulation scales Wang's design rules to a water-based, microfluidic regime:**

| Parameter | Wang et al. (2019) | Our FDTD simulation |
|-----------|-------------------|---------------------|
| Medium | Air ($c = 343$ m/s) | Water ($c = 1500$ m/s) |
| Frequency | 3.43 kHz | 1 MHz |
| Wavelength $\lambda$ | 100 mm | 1.5 mm |
| Channel width | 0.3$\lambda$ = 30 mm | 0.3$\lambda$ = 0.45 mm |
| Gate length | 0.6$\lambda$ = 60 mm | 0.6$\lambda$ = 0.9 mm |
| Phase control | Helmholtz resonator unit cells | Direct source phase difference |
| Threshold | 0.4 Pa (absolute) | Relative / normalised |

**Current grid:** dx = 50 μm, c = 1500 m/s (water)

| Frequency | $\lambda$ (mm) | 0.3$\lambda$ (Wang width) | $\lambda$/dx (pts/$\lambda$) | 0.3$\lambda$/dx (cells across channel) |
|-----------|---------------|---------------------------|----------------------------|----------------------------------------|
| 500 kHz   | 3.0           | 0.9 mm                    | 60                         | 18                                     |
| 750 kHz   | 2.0           | 0.6 mm                    | 40                         | 12                                     |
| **1 MHz** | **1.5**       | **0.45 mm**               | **30**                     | **9**                                  |
| 2 MHz     | 0.75          | 0.225 mm                  | 15                         | 4.5                                    |
| 3 MHz     | 0.5           | 0.15 mm                   | 10                         | 3                                      |

### Resolution Assessment

- **Comfortable** ($\geq$10 cells across channel): $\leq$ 1 MHz with dx = 50 μm
- **Acceptable** (6–9 cells): ~1 MHz — fine for proof-of-concept
- **Marginal** (3–5 cells): 2 MHz — geometry poorly resolved
- **Under-resolved** (<3 cells): $\geq$ 3 MHz — unreliable

### Recommended Design Point (dx = 50 μm)

- **Operating frequency**: **~1 MHz**
- **Wavelength**: $\lambda \approx 1.5$ mm
- **Inlet channel width**: $w \approx 0.3\lambda \approx$ **0.45 mm** (9 grid cells)
- **Gate length** (interference region): $L \approx 0.6\lambda \approx$ **0.9 mm**
- **Outlet width**: can match inlet or be slightly wider
- **Phase difference for logic**:
  - $\varphi = 0^\circ$ $\rightarrow$ constructive interference $\rightarrow$ **high amplitude** $\rightarrow$ logic "1" / AND
  - $\varphi = 180^\circ$ $\rightarrow$ destructive interference $\rightarrow$ **near-zero amplitude** $\rightarrow$ logic "0" / XOR

---

## References to Add to Thesis Bibliography

```bibtex
@article{wang2019binary,
  author  = {Wang, Yin and Xia, Jian-ping and Sun, Hong-xiang and Yuan, Shou-qi and Liu, Xiao-jun},
  title   = {Binary-phase acoustic passive logic gates},
  journal = {Scientific Reports},
  year    = {2019},
  volume  = {9},
  number  = {1},
  pages   = {8355},
  doi     = {10.1038/s41598-019-44769-0}
}
```

---

## Open Questions / TODO

- [x] Fix: Wang et al. frequency is **3.43 kHz** (not 5.8 kHz); wavelength is **100 mm** in air (not 59 mm).
- [ ] Decide: keep dx = 50 μm @ 1 MHz, or refine grid for higher frequency?
- [ ] Choose exact Y-junction geometry: symmetric or asymmetric? angled inlets or straight?
- [ ] How to implement phase-delay "unit cells" in FDTD? (obstacle arrays? local c(x,y) variation?)
- [ ] Define quantitative logic contrast threshold (Wang uses 0.4 Pa absolute — we need a relative / normalised metric)
