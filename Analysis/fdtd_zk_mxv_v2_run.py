#!/usr/bin/env python3
"""
fdtd_zk_mxv_v2_run.py
=====================
Run the 2-D ZK signed 2x2 MxV optimisation and save the result.
"""

import numpy as np
import pickle
from pathlib import Path

from fdtd_zk_mxv_v2 import optimise, loss_function, plot_result


W_target = np.array([
    [0.60, -0.40],
    [-0.20, 0.80]
])

print("Target matrix:", flush=True)
print(W_target, flush=True)
print("\nStarting optimisation ...", flush=True)

result = optimise(W_target, maxiter=20, popsize=4, workers=1, seed=42)

print("\nOptimisation finished.", flush=True)
print(f"Best MSE: {result.fun:.6f}", flush=True)

_, A, A_real = loss_function(result.x, W_target, return_details=True)
print("\nAchieved real matrix A:", flush=True)
print(A_real, flush=True)
print("\nAchieved complex matrix A:", flush=True)
print(A, flush=True)

plot_result(W_target, A_real, result.x)

# Save result
outdir = Path('results')
outdir.mkdir(exist_ok=True)
outpath = outdir / 'fdtd_zk_mxv_v2_result.pkl'
with open(outpath, 'wb') as f:
    pickle.dump({
        'W_target': W_target,
        'A': A,
        'A_real': A_real,
        'params': result.x,
        'mse': result.fun,
        'result': result,
    }, f)
print(f"\nSaved result to {outpath}", flush=True)
