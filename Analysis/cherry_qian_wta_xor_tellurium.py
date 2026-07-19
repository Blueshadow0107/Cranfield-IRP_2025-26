"""
Cherry-Qian-style winner-take-all XOR in Antimony/Tellurium.

Architecture:
- Pinned inputs X1, X2 (boundary species).
- Learned weights are initial concentrations of real molecular species
  W11, W12, W21, W22, R1, R2 that are consumed during a forward pass.
- Hidden species H1, H2 compete via annihilation; output Y is produced
  from the winning hidden via a second weighting reaction.

The ODE system is integrated by libRoadRunner.  We train only the
initial concentrations of the weight species with L-BFGS-B to minimise
the XOR loss on the final Y concentration.
"""

import numpy as np
import tellurium as te
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Antimony model
# ---------------------------------------------------------------------------
ANTIMONY_MODEL = """
model wta_xor
    // Boundary (pinned) inputs.  Their values are overwritten from Python.
    species X1, X2;
    X1 = 0; X2 = 0;

    // Learned weights: these are real species whose initial concentrations
    // are optimised.  They are consumed when they catalyse a reaction.
    species W11, W12, W21, W22;
    species R1, R2;

    // Dynamic species
    species H1, H2, Y;

    // Input -> hidden layer (weights are consumed)
    J1: X1 + W11 -> X1 + H1; k * X1 * W11;
    J2: X2 + W12 -> X2 + H1; k * X2 * W12;
    J3: X1 + W21 -> X1 + H2; k * X1 * W21;
    J4: X2 + W22 -> X2 + H2; k * X2 * W22;

    // Winner-take-all competition
    J5: H1 + H2 -> ; k_annih * H1 * H2;

    // Hidden -> output layer (readout weights are also consumed)
    J6: H1 + R1 -> H1 + Y; k_out * H1 * R1;
    J7: H2 + R2 -> H2 + Y; k_out * H2 * R2;

    // First-order degradation/dilution
    J8: H1 -> ; d * H1;
    J9: H2 -> ; d * H2;
    J10: Y -> ; d * Y;

    // Kinetic parameters
    k = 1.0;
    k_annih = 1.0;
    k_out = 1.0;
    d = 0.1;
end
"""

# XOR truth table: (X1, X2) -> target Y
XOR_INPUTS = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
], dtype=float)
XOR_TARGETS = np.array([0.0, 1.0, 1.0, 0.0], dtype=float)

WEIGHT_NAMES = ["W11", "W12", "W21", "W22", "R1", "R2"]
T_FINAL = 20.0
N_STEPS = 200


def load_model_with_weights(weights: np.ndarray):
    """Return a fresh roadrunner model with the given initial concentrations."""
    r = te.loada(ANTIMONY_MODEL)
    for name, value in zip(WEIGHT_NAMES, weights):
        r[name] = float(value)
    return r


def evaluate_xor(weights: np.ndarray) -> np.ndarray:
    """
    Run all four XOR cases and return the final Y concentration for each.

    The weight species are consumed during each simulation, so we start a
    fresh model (same initial weights) for every input case.
    """
    outputs = np.empty(4, dtype=float)
    for i, (x1, x2) in enumerate(XOR_INPUTS):
        r = load_model_with_weights(weights)
        r.X1 = x1
        r.X2 = x2
        result = r.simulate(0, T_FINAL, N_STEPS)
        outputs[i] = result["[Y]"][-1]
    return outputs


def loss_and_grad(weights: np.ndarray):
    """Return scalar MSE loss and its gradient w.r.t. initial weights."""
    w = np.asarray(weights, dtype=float)
    eps = np.sqrt(np.finfo(float).eps)

    outputs = evaluate_xor(w)
    residual = outputs - XOR_TARGETS
    loss = float(np.mean(residual ** 2))

    grad = np.empty_like(w)
    for j in range(len(w)):
        dw = np.zeros_like(w)
        dw[j] = eps * max(1.0, abs(w[j]))
        outputs_perturbed = evaluate_xor(w + dw)
        loss_perturbed = float(np.mean((outputs_perturbed - XOR_TARGETS) ** 2))
        grad[j] = (loss_perturbed - loss) / dw[j]

    return loss, grad


def train(seed: int = 0, max_iter: int = 200):
    """Run L-BFGS-B from a random initial guess."""
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0.1, 2.0, size=len(WEIGHT_NAMES))

    print("Initial weights:", dict(zip(WEIGHT_NAMES, x0)))
    print("Initial outputs:", evaluate_xor(x0))
    print("Initial loss:", loss_and_grad(x0)[0])

    result = minimize(
        loss_and_grad,
        x0,
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, None)] * len(WEIGHT_NAMES),
        options={"maxiter": max_iter, "disp": True},
    )

    final_weights = result.x
    final_outputs = evaluate_xor(final_weights)

    print("\nOptimization success:", result.success)
    print("Final loss:", result.fun)
    print("Final weights:", dict(zip(WEIGHT_NAMES, final_weights)))
    print("Final outputs:", final_outputs)
    print("Targets:        ", XOR_TARGETS)

    return result, final_outputs


if __name__ == "__main__":
    train()
