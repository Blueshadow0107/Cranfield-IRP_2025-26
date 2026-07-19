"""Diagnose the diffractive FDTD — visualise fields, traces, and baseline comparison."""
import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt
from diffractive_fdtd_jax import (
    run_fdtd, build_bc_mask, XOR_PATTERNS, XOR_TARGETS,
    NX, NY, DX, STEPS_PER_PERIOD, DETECTOR_X, DETECTOR_Y, N_DETECTORS
)
import jax.numpy as jnp

np.random.seed(42)
theta_rand = np.random.randn(100) * 1.0  # same init as training
theta_zero = np.zeros(100)                # no scatterers

def analyse(theta, label, nt=1200):
    bc = build_bc_mask(jnp.array(theta))
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"{label} — Field snapshots & detector traces")

    for idx in range(4):
        pattern = XOR_PATTERNS[idx]
        probes, field = run_fdtd(bc, pattern, nt=nt)
        probes = np.array(probes)
        field = np.array(field)

        # Field snapshot
        ax = axes[0, idx]
        vmax = np.abs(field).max()
        im = ax.imshow(field.T, origin='lower', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax,
                       extent=[0, NX*DX*1e3, 0, NY*DX*1e3])
        ax.set_title(f"Pat {idx}: ({int(np.argmax(XOR_TARGETS[idx]))})")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Detector traces
        ax = axes[1, idx]
        t_ms = np.arange(len(probes)) * (5e-5 * 0.5 / 1500) * 1e6  # µs
        for d in range(N_DETECTORS):
            ax.plot(t_ms, probes[:, d], label=f"Det {d}")
        ax.set_xlabel("Time (µs)")
        ax.set_ylabel("Pressure")
        ax.set_title(f"Det traces — Pat {idx}")
        ax.legend()
        ax.grid(True)

        # RMS in last 3 periods
        window = probes[-3*STEPS_PER_PERIOD:, :]
        rms = np.sqrt(np.mean(window**2, axis=0))
        print(f"  Pat {idx}: RMS = [{rms[0]:.5f}, {rms[1]:.5f}]")

    plt.tight_layout()
    fname = f"diagnose_{label.lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved {fname}\n")


print("=" * 60)
print("NO SCATTERERS (baseline)")
print("=" * 60)
analyse(theta_zero, "No scatterers")

print("=" * 60)
print("RANDOM SCATTERERS (N(0,1), ~50% ON)")
print("=" * 60)
analyse(theta_rand, "Random scatterers")
