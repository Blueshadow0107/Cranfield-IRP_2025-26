# Acoustic Energy and Length-Scale Derivation

**Date:** 27 June 2026  
**Purpose:** Clean, step-by-step derivation of the acoustic energy and length-scale estimates used to argue against the porous wave-computing direction. Intended as reference material for supervisor discussions and the final report.

## Source parameters

From `Analysis/fdtd_zk_mxv_v2.py`:

- Domain size: `L = 0.50 m`
- Frequency: `f = 2000 Hz`
- Sound speed in air: `c0 = 343 m/s`
- Source RMS amplitude: `p_rms = 1.0 Pa`
- Source position: `SRC_X = 0.08 m`
- Probe position: `PROBE_X = 0.42 m`
- Design region: `x = 0.15 m` to `0.35 m`
- Design grid: `3 x 3` blocks

## 1. Length scale

### Wavelength

```
lambda = c0 / f
       = 343 / 2000
       = 0.1715 m
       = 17.15 cm
```

### Geometry expressed in wavelengths

| Feature | Physical size | In wavelengths |
|---------|---------------|----------------|
| Full domain | 0.50 m | 2.9 lambda |
| Design region width | 0.20 m | 1.2 lambda |
| Each 3x3 design block in x | 0.067 m | 0.39 lambda |
| Source-to-probe distance | 0.34 m | 2.0 lambda |

### Interpretation

Diffractive computing needs many wavelengths to accumulate useful phase shifts. A device only ~3 wavelengths long, with design blocks nearly 0.4 wavelengths wide, lacks the spatial resolution to act as a programmable wave computer. Each block is too coarse to impose a controlled phase delay.

## 2. Energy scale

### Acoustic intensity

For a plane wave:

```
I = p_rms^2 / (rho0 * c0)
```

with `rho0 = 1.225 kg/m^3` and `c0 = 343 m/s`:

```
I = 1^2 / (1.225 * 343)
  = 1 / 420.175
  ≈ 2.38 x 10^-3 W/m^2
  ≈ 2.4 mW/m^2
```

### Energy through the device

Energy passing through area `A` in time `t`:

```
E = I * A * t
```

Take `A = 1 cm^2 = 10^-4 m^2` and `t = 1 ms = 10^-3 s`:

```
E = 2.4 x 10^-3 * 10^-4 * 10^-3
  = 2.4 x 10^-10 J
  = 0.24 nJ
```

So a single 1 ms acoustic operation injects roughly **0.24 nJ** into the device.

## 3. Useful vs. injected energy

Most of the wave energy propagates straight through. Only the interference contrast at the probes carries the result. If the probe intercepts 1% to 10% of the total power:

```
E_useful ≈ (0.01 to 0.10) * 0.24 nJ
         ≈ 2.4 pJ to 24 pJ
```

Even the optimistic useful-energy estimate is 10^2 to 10^4 times worse than a digital MAC operation (~1 fJ to 1 pJ) and far above a biological synapse (~0.1 to 1 fJ).

## 4. Why acoustic energy is fundamentally high

- Sound is a collective mechanical motion; creating even a 1 Pa wave requires moving many air molecules.
- 1 Pa is already quiet. Lowering pressure reduces SNR, not just energy.
- Higher frequencies give shorter wavelengths but stronger attenuation in porous media.
- The device must be filled with wave energy; computation cannot be localised to a tiny region.

## 5. Spatial degrees of freedom

Independent spatial modes scale roughly as:

```
N_modes ~ (L / lambda)^2
        ~ (0.50 / 0.1715)^2
        ~ 8.5
        ≈ 9
```

A small digital neural network has hundreds to thousands of weights. A 2D acoustic device at 2 kHz has only a handful of independent spatial channels, giving very low representational capacity.

## 6. Summary table

| Quantity | Value | Interpretation |
|----------|-------|----------------|
| Wavelength | 17 cm | Forces a large device |
| Domain length | ~3 lambda | Too short for diffractive computing |
| Design block size | ~0.39 lambda | Too coarse for phase control |
| Injected energy per op | ~0.24 nJ | 10^5 to 10^8 x digital MAC |
| Useful energy per op | ~2 to 24 pJ (optimistic) | Still 10^2 to 10^4 x digital |
| Spatial modes | ~9 | Very low capacity |

## 7. Implications for the pivot

These constraints persist even if the placeholder Zwikker-Kosten model is replaced by the correct Johnson-Champoux-Allard model:

- Long wavelength → large device.
- High energy per operation → poor efficiency.
- Few spatial modes → low computational capacity.

This is the physical justification for pivoting from acoustic wave computing to a reactive-fluid / CRN implementation of the Fluid Neural Network concept.

## References

- `Analysis/fdtd_zk_mxv_v2.py` — source parameters for the acoustic setup.
- Deck: `Presentations/wave-crn-feasibility-2026-06-27.pdf` — slide version of this argument.
