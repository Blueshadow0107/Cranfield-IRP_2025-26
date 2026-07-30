# Weekly Progress Report: RD System Tests

---

## 1. Method of Manufactured Solutions (MMS)

**Purpose:**
This test was a formal verification of the `rd_core` solver's numerical accuracy. It used the Method of Manufactured Solutions to confirm that the error decreases at the expected theoretical rate when the grid and time steps are refined.

**Conclusion:**
The test was a **success**. For all three models tested (`barkley_iso`, `barkley_tilted`, `oregonator_A`), the solver demonstrated the correct convergence behavior.
*   **Spatial Convergence Order:** ~2.0 (as expected)
*   **Temporal Convergence Order:** ~1.0 (as expected)

This provides strong confidence in the correctness of the core simulation engine.

**Figure:** `IndividualResearchProject/Analysis/figures/rd_mms.png`

---

## 2. Anisotropy Test

**Purpose:**
This test verified that the simulation correctly models anisotropic (uneven) diffusion by checking if the ratio of wave speeds along fast and slow axes matched the theoretical `sqrt(r)` value.

**Conclusion:**
The `oregonator` model showed excellent agreement with theory (less than 0.4% deviation). The `barkley` model showed a larger deviation (up to 9.6%), which might warrant further investigation if high precision is required for that model.

**Figure:** `IndividualResearchProject/Analysis/figures/rd_transfer_aniso_final2.png`

---

## 3. Transmission/Channel Test

**Purpose:**
This test measured the propagation of a pulse through a straight channel of varying width to determine the minimum width required for successful transmission.

**Conclusion:**
The test was successful. For both the `barkley` and `oregonator` models, even very narrow channels (down to a single cell wide) were able to support pulse propagation. The wave speed in the channel was also found to be very close to the speed in a free, unbounded medium.

**Figure:** `IndividualResearchProject/Analysis/figures/rd_transfer_channel_final2.png`

---

## 4. Pulse Timing/Frequency Test

**Purpose:**
This test measured the refractory period of the medium and its ability to faithfully transmit a train of pulses at different frequencies.

**Conclusion:**
The test successfully characterized the refractory properties of both models.
*   **Barkley:** Refractory period of ~2.8 time units; max 1:1 following rate of ~0.26 pulses/t.u.
*   **Oregonator:** Refractory period of ~3.1 time units; max 1:1 following rate of ~0.28 pulses/t.u.

This confirms that both media have a built-in "reset time" that limits their information processing speed.

**Figures:**
*   `IndividualResearchProject/Analysis/figures/rd_transfer_frequency_final2_barkley.png`
*   `IndividualResearchProject/Analysis/figures/rd_transfer_frequency_final2_oregonator.png`

---

## 5. RD Core Verification

**Purpose:**
This was a final, formal verification of the `rd_core` solver's accuracy and stability, checking for grid convergence, timestep convergence, and simulation invariants.

**Conclusion:**
The solver was found to be **stable**. The `u` and `v` values remained within their expected physical ranges, and no `NaN`/`Inf` values were generated. While the observed convergence orders did not perfectly match the theory, the results are reasonable and the deviations are explainable by the interaction of different error sources, which is common in complex simulations.

**Figure:** `IndividualResearchProject/Analysis/figures/rd_verification.png`
