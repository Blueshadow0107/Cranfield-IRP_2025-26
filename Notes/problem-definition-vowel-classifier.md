# Problem definition: Hughes-style vowel classifier in a 2D porous medium

This document is a clean, authoritative specification for the vowel-classification demonstration. It supersedes the exploratory discussion in `problem-definition-analog-wave-computer.md`.

---

## 1. Task

Reproduce the Hughes-style vowel-classification experiment (Hughes et al. 2019) in a 2D rigid-frame porous acoustic medium. The goal is to show that a trainable spatial map of porosity $\phi(x,y)$, and optionally flow resistivity $\sigma(x,y)$, can classify spoken vowels from time-integrated acoustic probe readouts.

Intended contribution: demonstrate that adding damping through $\sigma(x,y)$ gives extra design freedom beyond a pure sound-speed map.

---

## 2. Physical model

### 2.1 Assumptions

- Rigid porous skeleton, air-saturated pores.
- Frame motion neglected (rigid-frame equivalent-fluid model).
- Isothermal, linear acoustics.
- Two-dimensional domain.
- Simplified time-domain Zwikker-Kosten (ZK) model with piecewise-constant material properties.

### 2.2 Governing equations

Let $p(x,y,t)$ be acoustic pressure and $\mathbf{V}(x,y,t) = (V_x, V_y)$ be the macroscopic volume velocity. The simplified ZK equations are

$$
\frac{\rho_0 \alpha_\infty}{\phi} \frac{\partial \mathbf{V}}{\partial t} + \sigma \mathbf{V} = -\nabla p,
$$

$$
\frac{\partial p}{\partial t} + \rho_0 c_0^2 \nabla \cdot \mathbf{V} = 0.
$$

Parameters:

| Symbol | Meaning | Value |
|--------|---------|-------|
| $\rho_0$ | ambient air density | $1.21 \ \text{kg}/\text{m}^3$ |
| $c_0$ | ambient sound speed | $343 \ \text{m}/\text{s}$ |
| $\alpha_\infty$ | tortuosity | $1.0$ (fixed) |
| $\phi(x,y)$ | porosity | design variable, $0.3 \leq \phi \leq 1.0$ |
| $\sigma(x,y)$ | flow resistivity | design variable, $0 \leq \sigma \leq 10^5 \ \text{Pa}\cdot\text{s}/\text{m}^2$ |

The effective sound speed and characteristic impedance for the lossless limit ($\sigma = 0$) are

$$
c_\text{eff} = \frac{c_0}{\sqrt{\phi \alpha_\infty}}, \qquad
Z_\text{eff} = \frac{\rho_0 c_0 \sqrt{\alpha_\infty}}{\phi^{3/2}}.
$$

With $\alpha_\infty = 1$, this reduces to $c_\text{eff} = c_0 / \sqrt{\phi}$.

---

## 3. Domain

### 3.1 Geometry

The computational domain is the rectangle

$$
\Omega = [0, L_x] \times [0, L_y],
$$

with

$$
L_x = 0.80 \ \text{m}, \qquad L_y = 0.10 \ \text{m}.
$$

### 3.2 Grid

Uniform Cartesian grid with spacing

$$
\Delta x = \Delta y = 2.0 \ \text{mm},
$$

giving

$$
N_x = 400, \qquad N_y = 50.
$$

### 3.3 Boundaries

First-order absorbing boundary conditions on all four sides:

- $\Gamma_\text{left}$: $x = 0$
- $\Gamma_\text{right}$: $x = L_x$
- $\Gamma_\text{bottom}$: $y = 0$
- $\Gamma_\text{top}$: $y = L_y$

The absorbing condition is a plane-wave impedance match based on the local effective impedance. No PML or sponge layer is used in the baseline configuration.

---

## 4. Trainable design region

The trainable region occupies the middle third of the domain in $x$ and the full height in $y$:

$$
\Omega_\text{design} = \left[\frac{L_x}{3}, \frac{2L_x}{3}\right] \times [0, L_y].
$$

Numerically:

$$
\Omega_\text{design} = [0.2667 \ \text{m}, 0.5333 \ \text{m}] \times [0, 0.10 \ \text{m}].
$$

The region is split into a $4 \times 4$ block grid:

- 4 blocks along $x$, each of width $L_x/12 \approx 66.7 \ \text{mm}$.
- 4 blocks along $y$, each of height $L_y/4 = 25.0 \ \text{mm}$.

Within each block $(i,j)$ the material properties are uniform:

$$
\phi(x,y) = \phi_{ij}, \qquad \sigma(x,y) = \sigma_{ij}, \qquad (x,y) \in \text{block } (i,j).
$$

Total design variables: $2 \times 16 = 32$ (16 porosities + 16 resistivities).

---

## 5. Source and probes

### 5.1 Source

A pressure source is injected over a $3 \times 3$ grid-cell patch located just to the left of $\Omega_\text{design}$, centred vertically at $y = L_y/2$.

The injected pressure update is

$$
p(x,y,t) \leftarrow p(x,y,t) + s(t), \qquad (x,y) \in \Omega_\text{src}.
$$

The signal $s(t)$ is a normalised vowel waveform from the dataset.

### 5.2 Probes

Three passive pressure probes are placed near the right edge of the trainable region, at

$$
\mathbf{r}_1 = \left(x_\text{probe}, \frac{L_y}{4}\right), \qquad
\mathbf{r}_2 = \left(x_\text{probe}, \frac{L_y}{2}\right), \qquad
\mathbf{r}_3 = \left(x_\text{probe}, \frac{3L_y}{4}\right),
$$

with $x_\text{probe} \approx 2L_x/3$.

Numerically:

| Probe | $x$ | $y$ |
|-------|-----|-----|
| 1 | $0.5333 \ \text{m}$ | $0.025 \ \text{m}$ |
| 2 | $0.5333 \ \text{m}$ | $0.050 \ \text{m}$ |
| 3 | $0.5333 \ \text{m}$ | $0.075 \ \text{m}$ |

---

## 6. Readout and classifier

### 6.1 Probe energy

For each vowel sample, the readout feature is the time-integrated squared pressure at each probe:

$$
E_k = \int_{t_0}^{t_0 + T_\text{int}} p(\mathbf{r}_k, t)^2 \, dt, \qquad k = 1, 2, 3.
$$

The integration window $T_\text{int}$ is fixed and will be set from a baseline transit test once the solver is running.

### 6.2 Classifier

A trainable linear layer maps the three probe energies to class logits:

$$
\mathbf{z} = \mathbf{W} \mathbf{E} + \mathbf{b},
$$

with $\mathbf{E} = [E_1, E_2, E_3]^T$. The classifier weights $\mathbf{W}$ and $\mathbf{b}$ are trained jointly with, or separately from, the physical design variables $\phi_{ij}$ and $\sigma_{ij}$.

### 6.3 Loss

Cross-entropy against the vowel-class label:

$$
\mathcal{L} = -\sum_c y_c \log\left( \frac{e^{z_c}}{\sum_{c'} e^{z_{c'}}} \right).
$$

---

## 7. Dataset and validation

- Dataset: Peterson-Barney or equivalent vowel dataset, following Hughes et al. (2019).
- Classes: three vowel classes (e.g. /ae/, /i/, /u/).
- Samples: $\sim 279$ utterances from $\sim 93$ speakers, as in Hughes.
- Input normalisation: each vowel sample is normalised to the same amplitude before injection.
- Validation: 5-fold cross-validation.
  - Split the data into five groups of roughly equal size.
  - Train on four groups, test on the fifth.
  - Repeat so each group is the test fold once.
  - Reported accuracy is the mean over the five test folds.

---

## 8. Numerical parameters

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Domain length | $L_x$ | $0.80 \ \text{m}$ |
| Domain height | $L_y$ | $0.10 \ \text{m}$ |
| Grid spacing | $\Delta x = \Delta y$ | $2.0 \ \text{mm}$ |
| Grid points | $N_x \times N_y$ | $400 \times 50$ |
| Design region | $\Omega_\text{design}$ | middle third in $x$, full height in $y$ |
| Design blocks | — | $4 \times 4 = 16$ |
| Minimum wavelength | $\lambda_\text{min}$ | $\approx 86 \ \text{mm}$ at $4 \ \text{kHz}$ in free air |
| Grid points per min wavelength | — | $\approx 43$ |
| Time step | $\Delta t$ | fixed, set by fastest $c_\text{eff}$ in design space |
| Integration window | $T_\text{int}$ | to be set from transit test |

The CFL condition is

$$
\Delta t \leq \frac{\Delta x}{c_\text{eff,max} \sqrt{2}},
$$

with $c_\text{eff,max} = c_0 / \sqrt{\phi_\text{min}} = 343 / \sqrt{0.3} \approx 626 \ \text{m}/\text{s}$.

---

## 9. Model validity

The simplified time-domain ZK model is strictly valid when the pore radius is smaller than the viscous skin depth. At $4 \ \text{kHz}$ the skin depth is approximately $35 \ \mu\text{m}$, so the target pore radius is $10\text{--}20 \ \mu\text{m}$.

If optimisation produces designs with larger effective pores, they must be validated against the full ZK Helmholtz model or direct pore-resolving simulation.

---

## 10. Open questions

1. Should $\phi$ and $\sigma$ be treated as independent design variables, or linked through a pore-radius model such as $\sigma \approx 8\eta / (R^2 \phi)$?
2. Should the absorbing boundaries be upgraded to a PML or sponge layer?
3. Should $\sigma$ be included from the start, or added only after a pure-$\phi$ baseline is established?
4. What is the optimal integration window $T_\text{int}$?

---

## 11. References

- Hughes et al. (2019), "Wave physics as an analog recurrent neural network," *Science Advances*.
- Zwikker & Kosten (1949), *Sound Absorbing Materials*.
- Johnson, Champoux & Allard (1987), "Generalized law for the dynamic tortuosity," *Journal of Applied Physics*.
