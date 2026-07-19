"""
WTA XOR with symmetry-broken init and hinge loss.

The simplified WTA has degenerate minima where one hidden is turned off.
This script breaks symmetry by initialising the cross weights (W12, W21)
large and the same-input weights (W11, W22) small, matching the XOR
structure we want.
"""
import numpy as np
import tellurium as te
from scipy.optimize import minimize
from cherry_qian_wta_xor_tellurium import (
    ANTIMONY_MODEL, WEIGHT_NAMES, evaluate_xor, XOR_TARGETS
)


def hinge_loss_and_grad(weights, margin=0.1):
    """
    Hinge-style loss: penalise outputs that are on the wrong side of
    a margin around the target (0 or 1).
    """
    w = np.asarray(weights, dtype=float)
    eps = np.sqrt(np.finfo(float).eps)

    outputs = evaluate_xor(w)
    # For target 1: loss if output < 1 - margin
    # For target 0: loss if output > margin
    loss_vec = np.where(
        XOR_TARGETS > 0.5,
        np.maximum(0.0, 1.0 - margin - outputs),
        np.maximum(0.0, outputs - margin),
    )
    loss = float(np.mean(loss_vec ** 2))

    grad = np.empty_like(w)
    for j in range(len(w)):
        dw = np.zeros_like(w)
        dw[j] = eps * max(1.0, abs(w[j]))
        outputs_p = evaluate_xor(w + dw)
        loss_vec_p = np.where(
            XOR_TARGETS > 0.5,
            np.maximum(0.0, 1.0 - margin - outputs_p),
            np.maximum(0.0, outputs_p - margin),
        )
        loss_p = float(np.mean(loss_vec_p ** 2))
        grad[j] = (loss_p - loss) / dw[j]

    return loss, grad


def train_symmetry_broken(seed=42, max_iter=300):
    rng = np.random.default_rng(seed)
    # Cross weights large, same-input weights small
    x0 = np.array([
        0.2 + rng.uniform(0.0, 0.2),   # W11 (X1 -> H1)
        1.0 + rng.uniform(0.0, 0.5),   # W12 (X2 -> H1)
        1.0 + rng.uniform(0.0, 0.5),   # W21 (X1 -> H2)
        0.2 + rng.uniform(0.0, 0.2),   # W22 (X2 -> H2)
        1.0 + rng.uniform(0.0, 0.5),   # R1
        1.0 + rng.uniform(0.0, 0.5),   # R2
    ])

    print("Initial weights:", dict(zip(WEIGHT_NAMES, x0)))
    print("Initial outputs:", evaluate_xor(x0))

    result = minimize(
        hinge_loss_and_grad,
        x0,
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, None)] * len(WEIGHT_NAMES),
        options={"maxiter": max_iter},
    )

    outputs = evaluate_xor(result.x)
    print("\nFinal loss:", result.fun)
    print("Final weights:", dict(zip(WEIGHT_NAMES, result.x)))
    print("Final outputs:", outputs)
    print("Targets:", XOR_TARGETS)
    return result, outputs


if __name__ == "__main__":
    train_symmetry_broken()
