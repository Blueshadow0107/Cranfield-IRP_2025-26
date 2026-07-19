# Zwikker-Kosten derivation worksheet

Date: 2026-06-14
Status: work in progress — fill in the blanks yourself

---

## References to consult

### 1. COMSOL documentation (authoritative implementation)
- **Source:** COMSOL Acoustics Module User's Guide, "Zwikker-Kosten" section
- **URL:** https://doc.comsol.com/6.2/doc/com.comsol.help.aco/AcousticsModuleUsersGuide.pdf
- **Key point:** Defines rigid-frame ZK by complex density `rho_rig = rho_f / eps_p * [Bessel function term]` and complex bulk modulus `K = gamma p_A / eps_p * [Bessel function term]`. In the high-frequency / lossless limit this reduces to `rho_eff = rho0 alpha_inf / phi` and `K_eff = K0 / phi`.

### 2. SAPEM 2005 proceedings — lossless 1D ZK with variable tortuosity
- **Source:** "High frequency acoustical pulses travel through a slab of macroscopically-heterogeneous lossless air-filled porous material", Proceedings of SAPEM 2005
- **URL:** https://sapem2005.matelys.com/proceedingsSAPEM2005.pdf
- **Key equations (1D):**
  ```
  rho0 alpha(x) phi(x) ∂v/∂t = -∂(p phi)/∂x
  K_a^{-1} ∂p/∂t = -∂v/∂x
  ```
  For uniform `phi` this gives `c = c0 / sqrt(alpha)`.

### 3. Leissing PhD thesis — ZK impedance and causality
- **Source:** T. Leissing, "Numerical modeling of outdoor sound propagation and absorption by porous ground surfaces"
- **URL:** https://theses.hal.science/tel-00455590/file/65_Leissing.pdf
- **Key equations (1D):**
  ```
  -∂_x v = (Omega0/(rho0 c0^2)) ∂_t p'
  (Phi/Omega0) rho0 ∂_t v + sigma0 v = -∂_x p'
  ```
  Characteristic impedance (lossy):
  ```
  Z_c = rho0 c0 sqrt(Phi/Omega0^2 + i sigma0/(Omega0 rho0 omega))
  ```

### 4. IIAV review — high-frequency limits and physical meaning of tortuosity
- **Source:** "A Review of Acoustical Methods for Porous Material Characterisation", International Journal of Acoustics and Vibration
- **URL:** https://www.iiav.org/ijav/content/volumes/22_2017_497501490091138/vol_1/1166_fullpaper_1601601490091476.pdf
- **Key point:** High-frequency limits are `rho_t ≈ rho0 alpha_inf / phi` and `C_p ≈ phi/(gamma P0)`, so `c0(omega) ≈ c0 / sqrt(alpha_inf)`.

### 5. Allard & Atalla textbook (cited everywhere)
- **Source:** J. F. Allard and N. Atalla, *Propagation of Sound in Porous Media: Modelling Sound Absorbing Materials*, 2nd ed., Wiley
- **Note:** This is the standard reference. If you can access it through Cranfield library, read Chapter 5.

### 6. Miki empirical model (useful for impedance convention check)
- **Source:** Miki model for fibrous materials
- **URL:** https://eprints.whiterose.ac.uk/id/eprint/178399/1/Ferina_2021.pdf
- **Key point:** Normalized characteristic impedance `Z_tilde = alpha_inf^0.5 / phi [1 + 0.070 (f/sigma)^-0.632] - i ...`. In the lossless limit `Z_c = rho0 c0 sqrt(alpha_inf) / phi`.

---

## Candidate formulation A: standard ZK in pore velocity

Variables: pressure `p`, pore-fluid particle velocity `u`.

Equations (lossless, `sigma = 0`):

```
(rho0 alpha_inf) ∂u/∂t = -grad(p)                     ... (A1)
∂p/∂t + (rho0 c0^2 / phi) div(u) = 0                  ... (A2)
```

**Your tasks:**

1. Assume a plane wave `p = p0 exp(i(omega t - k x))`, `u = u0 exp(i(omega t - k x))`. Substitute into (A1) and (A2).

2. Show that:

```
c_eff = ____________________________
```

3. Show that the pore-velocity impedance is:

```
Z_u = p/u = ____________________________
```

4. Define the bulk/Darcy velocity `V = phi u`. Show that:

```
Z_V = p/V = ____________________________
```

5. Does `phi` change the wave speed in this formulation? ____________________________

---

## Candidate formulation B: bulk-velocity form used in our scaffold

Variables: pressure `p`, bulk/Darcy velocity `V`.

Equations (lossless, `sigma = 0`):

```
(rho0 alpha_inf / phi) ∂V/∂t = -grad(p)               ... (B1)
∂p/∂t + rho0 c0^2 div(V) = 0                          ... (B2)
```

**Your tasks:**

1. Assume a plane wave `p = p0 exp(i(omega t - k x))`, `V = V0 exp(i(omega t - k x))`. Substitute into (B1) and (B2).

2. Show that:

```
c_eff = ____________________________
```

3. Show that the bulk-velocity impedance is:

```
Z_V = p/V = ____________________________
```

4. Does `phi` change the wave speed in this formulation? ____________________________

---

## Reflection at a single porosity step

A plane wave in medium 1 (`phi1`, `alpha1`) hits an interface with medium 2 (`phi2`, `alpha2`).

Pressure reflection coefficient (based on continuity of `p` and normal volume flux `phi u`):

```
R = (Z2 - Z1) / (Z2 + Z1)
```

**Your tasks:**

1. Which impedance should go into this formula: `Z_u`, `Z_V`, or something else? ____________________________

2. Write `R` explicitly for:
   - Formulation A: `R = ____________________________`
   - Formulation B: `R = ____________________________`

3. For `alpha1 = alpha2 = 1`, `phi1 = 1.0`, `phi2 = 0.5`, compute `|R|`:
   - Formulation A: `|R| = ____________________________`
   - Formulation B: `|R| = ____________________________`

---

## Damping with flow resistivity

Add `sigma` back into the bulk-velocity momentum equation:

```
(rho0 alpha_inf / phi) ∂V/∂t + sigma V = -grad(p)
∂p/∂t + rho0 c0^2 div(V) = 0
```

**Your tasks:**

1. Derive the complex wavenumber `k(omega)` for a plane wave.

2. Identify the real part `k_r` and the imaginary part `k_i`.

3. Spatial attenuation coefficient (Np/m): `alpha_att = |k_i| = ____________________________`

4. Temporal damping rate for a travelling wave packet: `gamma = ____________________________`

5. For `phi = 1`, `alpha_inf = 1`, `sigma = 5000 Pa·s/m²`, `rho0 = 1.225 kg/m³`, `c0 = 343 m/s`, compute:
   - `alpha_att = ____________________________ Np/m`
   - `gamma = ____________________________ Np/s`

---

## Boundary conditions for the FDTD solver

We use first-order absorbing boundary conditions on all four sides. For the bulk-velocity variable `V`, the condition is:

```
p = ± Z_V V_n
```

where `V_n` is the normal bulk velocity and the sign depends on propagation direction.

**Your tasks:**

1. For Formulation A rewritten in bulk velocity, what is `Z_V`? ____________________________

2. For Formulation B, what is `Z_V`? ____________________________

3. Which `Z_V` should our code use if we commit to Formulation B? ____________________________

---

## Decision checklist

Before talking to your professors, decide:

- [ ] Which formulation (A or B) are we defending?
- [ ] What is the physical justification for that choice?
- [ ] Is `alpha_inf` fixed, or is it a function of `phi`?
- [ ] If `alpha_inf` is fixed at 1, does `phi` give us a useful sound-speed contrast?
- [ ] How do we validate the chosen formulation against analytic reflection/transmission?

---

## Notes

_Add your own notes and derivations below as you work through the worksheet._
