# Thermoviscous Neural Network — Null Findings
## Date: 2026-06-07

### What was tested
Batch sweep of 12 heat patterns across 8 heated strips (2×4 grid) in a 10 mm × 2.5 mm channel.
Physical model: 2D incompressible Navier-Stokes (streamfunction-vorticity) with Arrhenius viscosity ν(T) and conservative variable-ν diffusion.

### Parameter ranges
- Re = 10, 50, 100
- U_max = 4, 20, 40 mm/s
- Heat sources: 0–100 K/s per strip
- Temperature range: 300–347 K
- Viscosity range: 1.0×10⁻⁶ → 4.2×10⁻⁷ m²/s (58% reduction)

### Readout
2 Hz / 3 Hz amplitude ratio at outlet centerline (frequency mixing via nonlinear advection).

### Result
**The ratio is invariant to machine precision across all configurations.**

| Re | Patterns | Ratio | Amplitude scaling |
|----|----------|-------|-------------------|
| 10 | 12 | 1.710974046 | baseline |
| 50 | 2 | 1.710974 | 5× |
| 100 | 2 | 1.710974 | 10× |

All patterns (all-cold, all-warm, checkerboard, top-only, bottom-only, single strips, ramps) give the exact same ratio. Only `max_temp` and `min_nu` vary.

### Diagnosis
1. **Advection washes out thermal structure.** Pe ≈ 143–1430. Hot/cold patterns from 0.5 mm strips are homogenised downstream.
2. **Flow is too linear for differential frequency attenuation.** The 2 Hz and 3 Hz signals propagate independently; viscosity scales both proportionally.
3. **Ratio is a conserved quantity.** The centerline velocity is essentially 1D superposition; no mechanism for differential attenuation exists in straight channel flow.

### Conclusion
Viscosity modulation via local heating **does not** provide a trainable weight for frequency-ratio readout in straight 2D channel flow, Re = 10–100.

### Next direction
**Cylinder wake flow.** Place cylinder(s) in channel; read wake signal (vortex shedding frequency, recirculation length, or probe velocity). Cylinder wake is strongly Re-dependent; viscosity changes upstream of a cylinder should modulate vortex shedding, separation point, and wake structure — providing a nonlinear, trainable readout.
