# Zwikker–Kosten vs Johnson–Champoux–Allard Regime Decision

**Date:** 2026-06-18  
**Decision:** The 2 kHz acoustic-wave computing prototype is in the Johnson–Champoux–Allard (JCA) regime, not the Zwikker–Kosten (ZK) regime. ZK will be retained as a fast, tractable design proxy; JCA is the intended physical validation model.

## Operating conditions

From the current 2-D MxV benchmark (`Analysis/fdtd_zk_mxv_v2.py`):

- Frequency: `f = 2000 Hz`
- Angular frequency: `omega = 2 * pi * 2000 ≈ 12566 rad/s`
- Air properties: `rho0 = 1.225 kg/m³`, `eta = 1.81e-5 Pa·s`
- Design porosity range: `phi = 0.30 ... 0.95`
- Design flow-resistivity range: `sigma = 0 ... 1e5 Pa·s/m²`

The macroscopic design blocks are ~6.7 cm × 16.7 cm, but the relevant length scale for the regime check is the pore size `Lambda` (viscous characteristic length) of the material that would fill each block.

## Regime diagnostics

### Viscous penetration depth

```
delta_v = sqrt(2 * eta / (rho0 * omega)) ≈ 48.5 µm
```

`delta_v` is the thickness of the oscillatory viscous boundary layer in a pore.

### Womersley number

```
Wo = Lambda * sqrt(rho0 * omega / eta)
```

ZK is strictly valid only when `Wo << 1` (the boundary layer fills the pore and the velocity profile is parabolic).

| Pore size `Lambda` | Material example | `Wo` at 2 kHz | Regime |
|---|---:|---:|---|
| 30 µm | fine sintered / etched membrane | 0.87 | transitional, approaching ZK |
| 60 µm | stone wool | 1.75 | JCA / transitional |
| 100–150 µm | melamine, fiberglass, polyester foams | 2.9–4.4 | clearly JCA |

### Biot frequency

The ZK drag term `sigma * u` is frequency-independent only below the Biot frequency:

```
f_c = sigma * phi / (rho0 * alpha_inf) / (2*pi)
```

| Parameters | `f_c` | At 2 kHz |
|---|---:|---|
| `sigma=1e4`, `phi=0.99`, `alpha_inf=1.01` (melamine-like) | 1.27 kHz | above transition → JCA |
| `sigma=5e4`, `phi=0.95`, `alpha_inf=1.5` (damped design region) | 4.11 kHz | close to transition |

## Conclusion

At 2 kHz with typical porous materials (`Lambda ~ 100 µm`), `Wo ~ 3` and `f > f_c`. The flow is inertia-dominated with thin boundary layers, which is the JCA regime. The ZK assumptions (real, constant drag; uniform velocity profile across the pore) are not satisfied.

## Implications for the project

1. **Current ZK solver remains useful** as a fast inverse-design proxy. It gives a real-valued, frequency-independent approximation that is cheap to optimise.

2. **Physical interpretation must be cautious.** Optimised `(phi, sigma, alpha_inf)` sets from the ZK solver should not be claimed as physically exact for 2 kHz open-cell foams.

3. **Target validation model is JCA.** A time-harmonic Helmholtz solver with complex `rho_eff(omega)` and `K_eff(omega)` should eventually be used to re-evaluate promising ZK designs. This adds two parameters per block: viscous characteristic length `Lambda` and thermal characteristic length `Lambda'`.

4. **Manufacturing mapping** should use JCA parameter tables (porosity, resistivity, tortuosity, characteristic lengths) rather than ZK-only tables.

## Roadmap

1. Finish signed-weight exploration with the existing ZK FDTD proxy.
2. Build a frequency-domain JCA solver (Helmholtz in the design region) to check whether ZK-optimised designs survive in the correct physical regime.
3. If results differ significantly, move the optimiser to the JCA forward model.
4. Reserve OpenFOAM / COMSOL microstructure validation for a small number of final designs.

## References

- Johnson, D. L., Koplik, J., & Dashen, R. (1987). Theory of dynamic permeability and tortuosity in fluid-saturated porous media. *Journal of Fluid Mechanics*, 176, 379–402.
- Champoux, Y., & Allard, J. F. (1991). Dynamic tortuosity and bulk modulus in air-saturated porous media. *Journal of Applied Physics*, 70(4), 1975–1979.
- Zwikker, C., & Kosten, C. W. (1949). *Sound Absorbing Materials*. Elsevier.
