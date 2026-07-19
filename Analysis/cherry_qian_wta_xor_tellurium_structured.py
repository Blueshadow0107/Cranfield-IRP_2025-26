"""
WTA XOR with reparameterised input weights that enforce the XOR structure.

We learn a shared scale s and two asymmetries a, b:
    W11 = s - a
    W12 = s + a
    W21 = s + b
    W22 = s - b

This guarantees W11 + W12 = W21 + W22 = 2s, so the (1,1) case is
balanced and annihilates.  If a, b > 0 then:
    (0,1): H1 wins because W12 = s+a > W22 = s-b
    (1,0): H2 wins because W21 = s+b > W11 = s-a

R1 and R2 remain free.
"""
import numpy as np
import tellurium as te
from scipy.optimize import minimize
from cherry_qian_wta_xor_tellurium import ANTIMONY_MODEL, XOR_TARGETS

PARAM_NAMES = ["s", "a", "b", "R1", "R2"]


def weights_from_params(params):
    s, a, b, R1, R2 = params
    return np.array([
        max(0.0, s - a),   # W11
        max(0.0, s + a),   # W12
        max(0.0, s + b),   # W21
        max(0.0, s - b),   # W22
        max(0.0, R1),
        max(0.0, R2),
    ])


def evaluate_xor_from_params(params):
    from cherry_qian_wta_xor_tellurium import evaluate_xor
    return evaluate_xor(weights_from_params(params))


def mse_loss_and_grad(params):
    p = np.asarray(params, dtype=float)
    eps = np.sqrt(np.finfo(float).eps)

    outputs = evaluate_xor_from_params(p)
    residual = outputs - XOR_TARGETS
    loss = float(np.mean(residual ** 2))

    grad = np.empty_like(p)
    for j in range(len(p)):
        dp = np.zeros_like(p)
        dp[j] = eps * max(1.0, abs(p[j]))
        outputs_p = evaluate_xor_from_params(p + dp)
        loss_p = float(np.mean((outputs_p - XOR_TARGETS) ** 2))
        grad[j] = (loss_p - loss) / dp[j]

    return loss, grad


def train(seed=42, max_iter=300):
    rng = np.random.default_rng(seed)
    # s around 0.5, a and b positive and moderate
    x0 = np.array([
        0.5 + rng.uniform(-0.1, 0.1),   # s
        0.3 + rng.uniform(0.0, 0.2),    # a
        0.3 + rng.uniform(0.0, 0.2),    # b
        1.0 + rng.uniform(0.0, 0.5),    # R1
        1.0 + rng.uniform(0.0, 0.5),    # R2
    ])

    print("Initial params:", dict(zip(PARAM_NAMES, x0)))
    print("Initial weights:", dict(zip(["W11","W12","W21","W22","R1","R2"], weights_from_params(x0))))
    print("Initial outputs:", evaluate_xor_from_params(x0))

    result = minimize(
        mse_loss_and_grad,
        x0,
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.01, None), (0.0, None), (0.0, None), (0.0, None), (0.0, None)],
        options={"maxiter": max_iter},
    )

    outputs = evaluate_xor_from_params(result.x)
    print("\nFinal loss:", result.fun)
    print("Final params:", dict(zip(PARAM_NAMES, result.x)))
    print("Final weights:", dict(zip(["W11","W12","W21","W22","R1","R2"], weights_from_params(result.x))))
    print("Final outputs:", outputs)
    print("Targets:", XOR_TARGETS)
    return result, outputs


if __name__ == "__main__":
    train()
