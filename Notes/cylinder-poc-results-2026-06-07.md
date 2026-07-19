# Cylinder Array Proof of Concept — Results Summary
## Date: 2026-06-07

### Concept
Discrete cylinder configurations in channel flow act as "weights" that modulate the wake signal. Different arrangements produce different probe amplitudes, enabling binary separable outputs.

### Solver
`Analysis/cylinder_array_solver.py` — 2D incompressible NS (streamfunction-vorticity) with multiple square obstacles. Re=40, U_max=40 mm/s, DT=5e-5, T=1.0s.

### Key Results

| Configuration | Cylinders | Amplitude | Std dev | Notes |
|--------------|-----------|-----------|---------|-------|
| **none** | — | 6.0×10⁻⁶ | 5.5×10⁻⁹ | Baseline laminar flow |
| **single** | [100,25,10] | 7.7×10⁻³ | 6.2×10⁻⁶ | Standard vortex wake |
| **tandem_close** | [100,25,10], [115,25,10] | 9.5×10⁻³ | 8.5×10⁻⁶ | Inline close — enhanced |
| **tandem_far** | [100,25,10], [130,25,10] | 6.0×10⁻⁴ | 5.1×10⁻⁷ | Inline far — **suppressed** |
| **side_close** | [100,22,10], [100,28,10] | 1.0×10⁻² | 8.6×10⁻⁶ | Side-by-side — strongest |
| **side_far** | [100,20,10], [100,30,10] | 1.0×10⁻² | 8.8×10⁻⁶ | Side-by-side — strongest |

### Binary Separation
- **LOW**: none (6×10⁻⁶) or tandem_far (6×10⁻⁴)
- **HIGH**: side_far (1×10⁻²)
- **Ratio**: 1,700× between none and side_far
- **Threshold-friendly**: A sigmoid at ~5×10⁻³ cleanly separates LOW from HIGH

### Logic Gate Potential
| Input A | Input B | Geometry | Output | Gate |
|---------|---------|----------|--------|------|
| 0 | 0 | none | LOW | — |
| 1 | 0 | single | MED | — |
| 0 | 1 | single | MED | — |
| 1 | 1 | tandem_far | LOW | XOR-like |
| 1 | 1 | side_far | HIGH | AND-like |

### Thesis Framing
This is the **discrete proof of concept**. It establishes that:
1. Flow geometry modulates wake signal
2. Different configurations give distinct, separable outputs
3. A simple threshold (sigmoid) can extract binary logic

The next step is to generalize from discrete obstacles to **continuous porous media**, where the permeability field α(x,y) replaces on/off cylinders.

### Reference
Carzeni & Modarres-Sadeghi (2026) — collective behaviour of cylinder clusters in wakes.
Relevant finding: inline cylinders suppress shedding; side-by-side pairs enhance it.
