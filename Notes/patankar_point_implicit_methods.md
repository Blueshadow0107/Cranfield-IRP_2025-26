:
# Point-Implicit / Patankar Methods for Stiff ODEs

## 1. What "point implicit" means

In CFD and ODE terminology, **point implicit treatment** usually means: at each grid point or each ODE component, treat the local source term implicitly while leaving everything else explicit or in a simpler form. It is a cheap approximation to a fully implicit solve.

Classic example: if you have

```
dy/dt = F(y) + S(y)
```

where `F` is a (possibly non-stiff) flux and `S` is a stiff source, a point-implicit step looks like

```
y^{n+1} = y^n + dt * [ F(y^n) + S(y^{n+1}) ]
```

If `S(y)` is linear or can be linearized locally, `y^{n+1}` can be solved with a small (1x1 or block-diagonal) system rather than a global Jacobian.

## 2. Patankar's trick

Patankar (1980) introduced a way to keep source/sink terms positive in finite-volume transport equations. The idea:

- Split the source into a **production** part `p(y) >= 0` and a **destruction** part `d(y) >= 0`.
- Discretize production explicitly and destruction implicitly:

```
(y^{n+1} - y^n) / dt = p(y^n) - d(y^n) * (y^{n+1} / y^n)
```

Rearranging:

```
y^{n+1} = y^n * (1 + dt * p(y^n) / y^n) / (1 + dt * d(y^n) / y^n)
```

Because `p, d, y^n >= 0`, the updated value `y^{n+1}` is guaranteed non-negative. This is the **Patankar-Euler** scheme.

## 3. Modified Patankar schemes

The original Patankar scheme preserves positivity but does **not** conserve total mass in a closed system (production and destruction are treated asymmetrically).

**Modified Patankar-Euler (MPE)** fixes this by using a common denominator for all species in a conservation law. For a production-destruction system:

```
dy_i/dt = sum_j [ p_{ij}(y) - d_{ij}(y) ]
```

with `p_{ij} = d_{ji}` (conservation), the MPE scheme is:

```
y_i^{n+1} = y_i^n + dt * sum_j [ p_{ij}(y^n) - d_{ij}(y^n) * (y_i^{n+1} / y_i^n) * phi ]
```

where `phi` is a single Patankar weight chosen to enforce the conservation invariant. This makes the scheme:

- **positivity preserving** for any time step
- **conservative** (preserves linear invariants)
- **first-order accurate in time**

## 4. Modified Patankar Runge-Kutta (MPRK)

Higher-order extensions combine RK stages with Patankar weights. Common schemes:

| Scheme | Order | Properties |
|--------|-------|------------|
| MPE (Modified Patankar-Euler) | 1 | positive, conservative |
| MPRK22(a,b) | 2 | positive, conservative, parameter-dependent stability |
| MPRK(4,3) | 3 | positive, conservative |
| SSP-MPRK | 2/3 | strong-stability-preserving variants |

The key idea at each RK stage:

```
y_i^{(k)} = y_i^n + dt * sum_l a_{kl} * sum_j [ p_{ij}(y^{(l)}) - d_{ij}(y^{(l)}) * (y_i^{(k)} / y_i^{(l)}) * phi_{ij}^{(k)} ]
```

The Patankar weights `phi` are constructed so that positivity and conservation hold stage-by-stage.

## 5. How this applies to stiff chemical kinetics

For a CRN written in production-destruction form, MPRK schemes let you:

- Take time steps much larger than the fastest reaction timescale
- Avoid negative concentrations (unlike explicit RK)
- Preserve mass conservation exactly
- Avoid solving large nonlinear systems (unlike fully implicit BDF)

They are especially popular for:

- Biogeochemical models (NPZD, carbon cycles)
- Reactive Euler / combustion
- Chemical reaction networks with fast equilibria

## 6. Limitations and caveats

- **Stability vs positivity**: Positivity is unconditional, but oscillations around steady states can still occur unless the time step is small enough (Torlo et al. 2022).
- **Order reduction**: Near vanishing components, some MPRK schemes drop to first order.
- **Stiff transients**: Very stiff sources may still require small steps during initial transients.
- **Non-production-destruction terms**: Terms like `x * sum(alpha*Y)` in Dack's executive equation are not naturally a production-destruction pair, so you have to reformulate or split them.

## 7. Relevance to our Dack RNCRN

Dack's system is naturally a production-destruction system:

```
dX_i/dt = beta_i - X_i + X_i * sum_j alpha_{ij} Y_j
```

The first two terms are production/destruction. The last term is autocatalytic production if `alpha_{ij} > 0` and catalytic destruction if `alpha_{ij} < 0`. Both can be written as production-destruction pairs:

- `X_i + Y_j --alpha_{ij}--> 2 X_i + Y_j`  (production of X_i with Y_j as catalyst)
- `X_i + Y_j --|alpha_{ij}|--> Y_j`        (destruction of X_i with Y_j as catalyst)

For perceptrons:

```
mu * dY_j/dt = gamma + theta_j Y_j + Y_j * sum_i omega_{ji} X_i - Y_j^2
```

The terms `gamma`, `theta_j Y_j`, `Y_j * sum(omega X)` are production/destruction terms, and `-Y_j^2` corresponds to `2 Y_j --> Y_j`.

So Dack's RNCRN is exactly the kind of system MPRK schemes are designed for. Using MPRK could let us:

- Simulate with `mu` much smaller without crashing the explicit solver
- Train with larger numerical time steps
- Guarantee non-negative concentrations during training

## 8. Key references

| Reference | What it covers |
|-----------|----------------|
| Patankar (1980) | Original "Numerical Heat Transfer and Fluid Flow"; source-term discretization |
| Burchard et al. (2003, 2005) | Modified Patankar-Euler and MPRK22 for biogeochemical models |
| Burchard et al. (2005) "Application of modified Patankar schemes to stiff biogeochemical models" | Stiff NPZD test, comparison to explicit RK |
| Torlo, Offner, Ranocha (2022) | Issues with Patankar schemes: oscillations, order reduction |
| Izgin et al. (2023) | Stability analysis of SSP-MPRK schemes |
| Huang, Yang, Zhu (2024) | Bound-preserving DG + modified Patankar for reacting flows |

## 9. Bottom line

Point-implicit / Patankar methods are a family of **linearly implicit** time integrators that preserve positivity and conservation for production-destruction systems. They are well suited to stiff CRNs like Dack's RNCRN, and could replace our `LSODA` calls during training to make the full two-timescale system more robust.
