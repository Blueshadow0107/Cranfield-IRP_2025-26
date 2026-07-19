"""Diagnostic: simulate each XOR case with the trained weights and plot trajectories."""
import numpy as np
import matplotlib.pyplot as plt
import tellurium as te
from cherry_qian_wta_xor_tellurium import (
    ANTIMONY_MODEL, WEIGHT_NAMES, load_model_with_weights, XOR_INPUTS, XOR_TARGETS
)

# Use the trained weights from the last run
final_weights = np.array([
    23.768063568111806,
    0.20840117539467454,
    0.05513583356716048,
    0.07423392530758018,
    3.6381606842698226,
    3.420143781070516,
])

fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
axes = axes.ravel()

for ax, (x1, x2), target in zip(axes, XOR_INPUTS, XOR_TARGETS):
    r = load_model_with_weights(final_weights)
    r.X1 = x1
    r.X2 = x2
    result = r.simulate(0, 20, 200)
    t = result[:, 0]
    ax.plot(t, result["[H1]"], label="H1")
    ax.plot(t, result["[H2]"], label="H2")
    ax.plot(t, result["[Y]"], label="Y")
    ax.axhline(target, color="k", linestyle="--", alpha=0.5, label=f"target={target}")
    ax.set_title(f"X1={int(x1)}, X2={int(x2)}  ->  Y_final={result['[Y]'][-1]:.3f}")
    ax.set_xlabel("time")
    ax.set_ylabel("concentration")
    ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig("Analysis/cherry_qian_wta_xor_tellurium_diagnostic.png", dpi=150)
print("Saved Analysis/cherry_qian_wta_xor_tellurium_diagnostic.png")

# Also print the final Y for each case
print("\nFinal outputs:")
for (x1, x2), target in zip(XOR_INPUTS, XOR_TARGETS):
    r = load_model_with_weights(final_weights)
    r.X1 = x1
    r.X2 = x2
    y = r.simulate(0, 20, 200)["[Y]"][-1]
    print(f"  X1={int(x1)} X2={int(x2)}  target={target}  Y={y:.4f}")
