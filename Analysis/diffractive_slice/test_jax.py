"""Quick validation of JAX-based differentiable FDTD."""
import sys, time
sys.path.insert(0, '.')

import numpy as np
from diffractive_fdtd_jax import loss_fn, grad, build_bc_mask, run_fdtd, evaluate
import jax.numpy as jnp

np.random.seed(42)
theta = np.random.randn(100) * 0.1
theta_jax = jnp.array(theta)

nt = 1200
print("=" * 50)
print("JAX FDTD validation")
print(f"nt={nt}, grid=300x200, params=100")
print("=" * 50)

# 1. Forward pass
t0 = time.time()
print("\n1. Forward pass...", flush=True)
loss0 = float(loss_fn(theta_jax, nt=nt))
print(f"   Loss = {loss0:.4f}")
print(f"   Time = {time.time()-t0:.1f}s")

# 2. Gradient
t0 = time.time()
print("\n2. Gradient evaluation...", flush=True)
g = grad(loss_fn)(theta_jax, nt=nt)
g_np = np.array(g)
print(f"   |grad| = {np.linalg.norm(g_np):.4f}")
print(f"   finite = {np.all(np.isfinite(g_np))}")
print(f"   nonzero = {np.sum(np.abs(g_np) > 1e-10)}/100")
print(f"   Time = {time.time()-t0:.1f}s")

# 3. A few training steps
t0 = time.time()
print("\n3. Training 20 steps...", flush=True)
lr = 0.5
theta_cur = theta_jax
for step in range(20):
    g = grad(loss_fn)(theta_cur, nt=nt)
    theta_cur = theta_cur - lr * g
    loss = float(loss_fn(theta_cur, nt=nt))
    if step % 5 == 0:
        print(f"   Step {step:2d}: loss = {loss:.4f}")

print(f"\n   Final loss = {loss:.4f}")
print(f"   Training time = {time.time()-t0:.1f}s")

# 4. Evaluation
print("\n4. Evaluation...", flush=True)
evaluate(theta_cur, nt=nt)

print("\nDone.")
