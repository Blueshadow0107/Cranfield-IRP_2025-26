"""Full-domain validation — conservative memory usage."""
import sys, gc, time
sys.path.insert(0, '.')
from diffractive_fdtd import loss_fn, grad, evaluate
import numpy as np

np.random.seed(42)
theta = np.random.randn(100) * 0.1
nt = 500
lr = 0.5

print("="*50)
print("Full-domain gradient check")
print(f"nt={nt}, grid=300x200, params=100")
print("="*50)

t0 = time.time()
print("\n1. Single gradient evaluation...", flush=True)
g = grad(loss_fn)(theta, nt=nt)
print(f"   |grad| = {np.linalg.norm(g):.4f}")
print(f"   nonzero = {np.sum(np.abs(g) > 1e-10)}/100")
print(f"   time = {time.time()-t0:.1f}s")

t0 = time.time()
print(f"\n2. Training {30} iterations...", flush=True)
losses = []
for step in range(30):
    g = grad(loss_fn)(theta, nt=nt)
    theta = theta - lr * g
    loss = float(loss_fn(theta, nt=nt))
    losses.append(loss)
    if step % 5 == 0:
        print(f"   Step {step:2d}: loss = {loss:.4f}  |grad| = {np.linalg.norm(g):.4f}")
    gc.collect()

print(f"\n   Final loss = {losses[-1]:.4f}")
print(f"   Training time = {time.time()-t0:.1f}s")

print("\n3. Evaluation...", flush=True)
evaluate(theta, nt=nt)

# Save
np.save("theta_opt.npy", theta)
print("\nSaved theta_opt.npy")
