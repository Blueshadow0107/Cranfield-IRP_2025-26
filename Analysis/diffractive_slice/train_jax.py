"""Long training run with JIT-compiled gradient steps."""
import sys, time
sys.path.insert(0, '.')

import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit
from functools import partial
from diffractive_fdtd_jax import loss_fn, evaluate, plot_results

# JIT-compile the gradient. nt is static because jnp.arange needs a concrete value.
@partial(jit, static_argnums=(1,))
def _grad_fn(theta, nt):
    return grad(loss_fn, argnums=0)(theta, nt)

np.random.seed(42)
# Stronger init: ~half scatterers start as "on"
theta = jnp.array(np.random.randn(100) * 1.0)

nt = 1200
n_iter = 300
lr = 0.1       # gentler step for SGD
beta = 0.9     # momentum

# Momentum buffer
v = jnp.zeros_like(theta)

print("=" * 60, flush=True)
print("JAX Diffractive FDTD — SGD with Momentum", flush=True)
print("=" * 60, flush=True)
print(f"nt={nt}, grid=300x200, params=100, iterations={n_iter}", flush=True)
print(f"Optimizer: SGD + momentum (lr={lr}, beta={beta})", flush=True)
print(f"Init: N(0, 1.0)  (~50% scatterers ON)", flush=True)
print(f"JIT compilation: enabled", flush=True)
print(flush=True)

# Warm-up JIT compilation
print("Warming up JIT compilation...", flush=True)
t0 = time.time()
_ = float(loss_fn(theta, nt))
_ = _grad_fn(theta, nt)
print(f"   Compile time = {time.time()-t0:.1f}s\n", flush=True)

losses = []

print("Training...", flush=True)
t_start = time.time()
for it in range(1, n_iter + 1):
    g = _grad_fn(theta, nt)
    v = beta * v - lr * g
    theta = theta + v
    
    # Compute loss every 5 iters
    if it % 5 == 0 or it == 1:
        loss = float(loss_fn(theta, nt))
        losses.append(loss)
        elapsed = time.time() - t_start
        # Count how many scatterers are "on"
        n_on = int(jnp.sum(jnp.where(theta > 0, 1, 0)))
        print(f"Iter {it:4d}/{n_iter}  loss = {loss:.4f}  |g| = {float(jnp.linalg.norm(g)):.4f}  ON={n_on}/100  time = {elapsed:.1f}s", flush=True)

total = time.time() - t_start
print(f"\nTotal training time = {total:.1f}s  ({total/n_iter:.1f}s/iter)", flush=True)

# Final evaluation
print("\nFinal evaluation...", flush=True)
evaluate(theta, nt=nt)

# Save
np.save("theta_jax_opt.npy", np.array(theta))
np.save("losses_jax.npy", np.array(losses))
print("\nSaved theta_jax_opt.npy and losses_jax.npy", flush=True)

# Plot
print("\nPlotting...", flush=True)
plot_results(theta, losses, nt=nt)
