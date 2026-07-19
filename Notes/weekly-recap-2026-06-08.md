# Weekly Recap — 2026-06-05 to 2026-06-08

## What we did

### 1. Thermoviscous neural network — NULL FINDINGS (June 7)
- **Hypothesis**: Temperature-dependent viscosity `ν(T)` via local heating strips could act as trainable "weights"
- **Model**: 2D incompressible NS (streamfunction-vorticity) with Arrhenius viscosity + conservative energy equation
- **Test**: 12 heat patterns across 8 heated strips (2×4 grid), Re = 10–100, U_max = 4–40 mm/s
- **Readout**: 2 Hz / 3 Hz amplitude ratio at outlet centerline (frequency mixing)
- **Result**: **Ratio is invariant to machine precision across all configurations.**
  - All patterns (all-cold, all-warm, checkerboard, top-only, bottom-only, single strips, ramps) give ratio = 1.710974046
  - Only `max_temp` and `min_nu` vary; the frequency ratio is a conserved quantity
- **Diagnosis**:
  1. Advection washes out thermal structure (Pe ≈ 143–1430)
  2. Flow is too linear for differential frequency attenuation
  3. Centerline velocity is essentially 1D superposition
- **Conclusion**: Viscosity modulation via local heating **does not** provide trainable weights in straight 2D channel flow.
- **Pivot**: Cylinder wake flow — place cylinder(s) in channel; read wake signal. Cylinder wake is strongly Re-dependent.

**Files**: `thermoviscous_channel.py`, `thermoviscous_cylinder.py`, `sweep_heat_patterns.py`, `test_cylinder_heat.py`, `test_cylinder_heat_re20.py`, `test_high_re.py`

---

### 2. Cylinder array proof of concept — BINARY SEPARABLE OUTPUTS (June 7)
- **Solver**: `cylinder_array_solver.py` — 2D incompressible NS (streamfunction-vorticity) with multiple **square obstacles**
- **Config**: Re=40, U_max=40 mm/s, DT=5e-5, T=1.0s
- **Six configurations tested**: none, single, tandem_close, tandem_far, side_close, side_far
- **Readout**: Probe amplitude downstream of obstacle cluster

| Configuration | Amplitude | Std dev | Notes |
|--------------|-----------|---------|-------|
| **none** | 6.0×10⁻⁶ | 5.5×10⁻⁹ | Baseline laminar flow |
| **single** | 7.7×10⁻³ | 6.2×10⁻⁶ | Standard vortex wake |
| **tandem_close** | 9.5×10⁻³ | 8.5×10⁻⁶ | Inline close — enhanced |
| **tandem_far** | 6.0×10⁻⁴ | 5.1×10⁻⁷ | Inline far — **suppressed** |
| **side_close** | 1.0×10⁻² | 8.6×10⁻⁶ | Side-by-side — strongest |
| **side_far** | 1.0×10⁻² | 8.8×10⁻⁶ | Side-by-side — strongest |

- **Binary separation**: Ratio = 1,700× between none and side_far. A sigmoid at ~5×10⁻³ cleanly separates LOW from HIGH.
- **Logic gate potential**:
  - XOR-like: two inputs → tandem_far → LOW
  - AND-like: two inputs → side_far → HIGH
- **Thesis framing**: This is the **discrete proof of concept**. Flow geometry modulates wake signal; different configurations give distinct, separable outputs.

**Files**: `cylinder_array_solver.py`, `cylinder_array_brinkman.py`, `test_cylinder_array.py`, `test_cylinder_array_fast.py`, `test_cylinder_vortex_sweep.py`

---

### 3. Womersley / oscillatory inlet experiments (June 7)
- **Concept**: Pulsatile inlet flow `U(t) = U_max * (1 + ε·sin(2πft)) * parabolic(y)` past cylinder(s)
- **Tests**: 2 Hz, 3 Hz, and dual-frequency (2+3 Hz) modes
- **Solvers**: `womersley_cylinder.py`, `womersley_cylinder_dual.py`
- **Result**: Oscillatory inlet confirmed as viable excitation mechanism; dual-frequency input sets up frequency-mixing readout tests

**Files**: `womersley_cylinder.py`, `womersley_cylinder_dual.py`, `womersley_demo.py`, `test_womersley.py`, `test_womersley_sweep.py`

---

### 4. Face-consistent collocated projection solver — WORKING (June 7–8)
- **Evolution**: `primitive_ns_solver.py` → `collocated_ns_solver.py` → `projection_solver.py`
- **The fix**: Divergence and pressure gradient both defined at cell **faces**
- `div_face(grad_face(p)) == ∇²p` exactly — no stencil mismatch
- Ghost-cell formulation: arrays are `(nx+2, ny+2)`
- Stable for all tested configs: channel flow, single cylinder, tandem, side-by-side, grating

| Approach | Result |
|----------|--------|
| Cell-center gradient `(p[i+1]-p[i-1])/(2dx)` | NaN in 15 steps |
| Face gradient `(p[i+1]-p[i])/dx` | Stable for 10,000+ steps |

**Files**: `primitive_ns_solver.py`, `collocated_ns_solver.py`, `debug_collocated.py`, `projection_solver.py`, `projection_skeleton.py`, `test_refactored_solver.py`

---

### 5. Brinkman vs discrete ψ–ω comparison (June 8)
- Brinkman = smooth porous obstacles → narrower wake, less blockage
- Discrete ψ–ω = hard square walls → wider wake, more blockage
- Both capture recirculation but quantitative metrics differ by ~2×
- Brinkman solver is ~100× faster (vectorized NumPy vs nested Python loops)

**Files**: `compare_solvers_quantitative.py`, `run_brinkman_sweep.py`

---

### 6. Frequency sweep (oscillatory inlet) — LINEAR PASS-THROUGH (June 8)
- Tested ε=0.1 and ε=0.3 across 15 frequencies (0.5–20 Hz)
- **Result**: flow is essentially linear. Output tracks input perfectly.
- 2nd harmonics: 1% at ε=0.1, 3% at ε=0.3 — still negligible
- Single smooth obstacle at Re=100 is a "wire," not a "neuron"

**Files**: `frequency_transfer_sweep.py`, `frequency_transfer_sweep_eps0.3.py`

---

### 7. Reynolds number sweep (10–1000) — BRINKMAN SUPPRESSES SHEDDING (June 8)
- Brinkman porous cylinder **suppresses vortex shedding**
- Hard-wall cylinder sheds at Re≈47; Brinkman cylinder doesn't shed until Re≈720
- Mean outlet velocity ~constant across all Re (insensitive to viscosity)
- Outlet profile narrows with increasing Re (stronger wake deficit)

**Files**: `re_sweep.py`, `re_sweep_outlet_profile.py`

---

### 8. Porous grating test — INVISIBLE TO FLOW (June 8)
- Alternating high-α/low-α stripes perpendicular to flow
- **Result**: grating is invisible to flow at Re=100
- No frequency-dependent behavior, no harmonic generation
- Flow goes around stripes through gaps; smooth edges prevent separation

**Files**: `test_grating.py`, `grating_frequency_sweep.py`

---

### 9. Solver refactor (June 8)
- Extracted all solver machinery into `projection_solver.py`
- `ProjectionSolver` class with:
  - `build_alpha()` / `build_grating()` — obstacle geometry
  - `stable_dt()` — automatic CFL/viscous stability
  - `run()` — full simulation with recording callbacks
- New sweep scripts are ~30 lines instead of 200

---

### 10. Dashboard and visualization tools (June 7)
- `dashboard_server.py` — web server for browsing simulation result history
- Multiple HTML notes documenting solver evolution and code structure:
  - `solver-evolution.html` — timeline of solver versions v1–v7
  - `code-structure.html` — modular architecture diagram
  - `collocated_viewer.html` — interactive collocated solver visualization
  - `irp-code-study.html` — full code study and analysis
  - `temperature-equation-proposal.html` — thermoviscous equation derivation
  - `discrete-to-porous-literature.html` — literature mapping discrete→porous

---

## Key technical insight

The **face-consistent stencil** is the single most important fix. The incompatibility between `div(cell-center grad)` and `∇²p` was the root cause of all earlier solver failures.

## What we learned for the thesis

1. **Single smooth obstacles are too simple** for analog neural computation
2. **Brinkman porous media suppress nonlinear dynamics** — need sharper edges or higher Re
3. **Outlet mean velocity is a poor readout** — insensitive to obstacle config and Re
4. **Outlet profile shape** IS sensitive to Re and geometry — better readout candidate
5. **Frequency mixing requires either**: sharp obstacles, multiple interacting obstacles, or higher Re
6. **Thermoviscous effects do not work** in straight channel flow — advection homogenises temperature
7. **Discrete square cylinders DO produce separable outputs** — the discrete PoC is valid

## Open questions

- Can a porous wall with a sharp aperture create jet instability?
- Would multiple gratings in series create cavity resonance?
- Is Re=100 simply too low for interesting nonlinear behavior?
- Should we move to sharp-edged obstacles (step-function α) instead of smooth Brinkman?
- Can we bridge the discrete cylinder PoC to continuous porous media?

## Files created this week

**Solvers**: `primitive_ns_solver.py`, `collocated_ns_solver.py`, `projection_solver.py`, `projection_skeleton.py`, `cylinder_array_solver.py`, `cylinder_array_brinkman.py`, `thermoviscous_channel.py`, `thermoviscous_cylinder.py`, `womersley_cylinder.py`, `womersley_cylinder_dual.py`

**Tests & Sweeps**: `test_cylinder_array.py`, `test_cylinder_array_fast.py`, `test_cylinder_heat.py`, `test_cylinder_heat_re20.py`, `test_cylinder_vortex_sweep.py`, `test_cylinder_re_sweep.py`, `test_high_re.py`, `test_womersley.py`, `test_womersley_sweep.py`, `sweep_heat_patterns.py`, `compare_solvers_quantitative.py`, `run_brinkman_sweep.py`, `frequency_transfer_sweep.py`, `frequency_transfer_sweep_eps0.3.py`, `re_sweep.py`, `re_sweep_outlet_profile.py`, `test_grating.py`, `grating_frequency_sweep.py`, `test_refactored_solver.py`, `debug_collocated.py`

**Tools**: `dashboard_server.py`, `womersley_demo.py`

**Notes**: `thermoviscous-null-findings-2026-06-07.md`, `cylinder-poc-results-2026-06-07.md`, `solver-evolution.html`, `code-structure.html`, `collocated_viewer.html`, `irp-code-study.html`, `temperature-equation-proposal.html`, `discrete-to-porous-literature.html`
