"""
Multi-seed + regularized training for the Tellurium WTA XOR model.
"""
import numpy as np
import tellurium as te
from scipy.optimize import minimize
from cherry_qian_wta_xor_tellurium import (
    ANTIMONY_MODEL, WEIGHT_NAMES, evaluate_xor, XOR_TARGETS
)


def regularized_loss_and_grad(weights, alpha=0.01):
    w = np.asarray(weights, dtype=float)
    eps = np.sqrt(np.finfo(float).eps)

    outputs = evaluate_xor(w)
    residual = outputs - XOR_TARGETS
    data_loss = np.mean(residual ** 2)
    reg_loss = alpha * np.mean(w ** 2)
    loss = float(data_loss + reg_loss)

    grad = np.empty_like(w)
    for j in range(len(w)):
        dw = np.zeros_like(w)
        dw[j] = eps * max(1.0, abs(w[j]))
        outputs_perturbed = evaluate_xor(w + dw)
        data_loss_p = np.mean((outputs_perturbed - XOR_TARGETS) ** 2)
        reg_loss_p = alpha * np.mean((w + dw) ** 2)
        loss_p = data_loss_p + reg_loss_p
        grad[j] = (loss_p - loss) / dw[j]

    return loss, grad


def train_one(seed, alpha=0.01, max_iter=300):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0.1, 2.0, size=len(WEIGHT_NAMES))
    result = minimize(
        regularized_loss_and_grad,
        x0,
        args=(alpha,),
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, None)] * len(WEIGHT_NAMES),
        options={"maxiter": max_iter},
    )
    return result.fun, result.x, result.success


if __name__ == "__main__":
    best = None
    for seed in range(10):
        loss, weights, success = train_one(seed)
        outputs = evaluate_xor(weights)
        print(f"seed={seed:2d} loss={loss:.6f} success={success}")
        print(f"  weights={dict(zip(WEIGHT_NAMES, np.round(weights, 3)))}")
        print(f"  outputs={np.round(outputs, 3)}  targets={XOR_TARGETS}")
        if best is None or loss < best[0]:
            best = (loss, weights, seed)

    print("\nBest seed:", best[2])
    print("Best loss:", best[0])
    print("Best weights:", dict(zip(WEIGHT_NAMES, best[1])))
    print("Best outputs:", evaluate_xor(best[1]))
