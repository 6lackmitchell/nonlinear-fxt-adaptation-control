"""system.py

Defines functions describing the system dynamics for the quadrotor
dynamic 6 degree-of-freedom model.

"""
import builtins
from importlib import import_module
import symengine as se
import numpy as np
from nptyping import NDArray

from core.dynamics_wrappers import (
    dyn_wrapper,
    control_affine_system_deterministic,
    control_affine_system_stochastic,
    first_order_forward_euler,
)

vehicle = builtins.PROBLEM_CONFIG["vehicle"]
control_level = builtins.PROBLEM_CONFIG["control_level"]
situation = builtins.PROBLEM_CONFIG["situation"]
mod = "models." + vehicle + "." + control_level + "." + situation

# Programmatic version of 'from mod import *'
module = import_module(mod + ".physical_params")
globals().update(
    {n: getattr(module, n) for n in module.__all__}
    if hasattr(module, "__all__")
    else {k: v for (k, v) in module.__dict__.items() if not k.startswith("_")}
)
module = import_module(mod + ".timing_params")
globals().update(
    {n: getattr(module, n) for n in module.__all__}
    if hasattr(module, "__all__")
    else {k: v for (k, v) in module.__dict__.items() if not k.startswith("_")}
)

# Define Symbolic State
xs = se.symbols(["n", "e", "d", "u", "v", "w", "phi", "theta", "psi", "p", "q", "r"])

# Define symbolic system dynamics
f_symbolic = se.DenseMatrix(
    [
        xs[3] * (se.cos(xs[7]) * se.cos(xs[8]))
        + xs[4] * (se.sin(xs[6]) * se.sin(xs[7]) * se.cos(xs[8]) - se.cos(xs[6]) * se.sin(xs[8]))
        + xs[5] * (se.cos(xs[6]) * se.sin(xs[7]) * se.cos(xs[8]) + se.sin(xs[6]) * se.sin(xs[8])),
        xs[3] * (se.cos(xs[7]) * se.sin(xs[8]))
        + xs[4] * (se.sin(xs[6]) * se.sin(xs[7]) * se.sin(xs[8]) + se.cos(xs[6]) * se.cos(xs[8]))
        + xs[5] * (se.cos(xs[6]) * se.sin(xs[7]) * se.sin(xs[8]) - se.sin(xs[6]) * se.cos(xs[8])),
        xs[3] * (se.sin(xs[7]))
        - xs[4] * (se.sin(xs[6]) * se.cos(xs[7]))
        - xs[5] * (se.cos(xs[6]) * se.cos(xs[7])),
        xs[11] * xs[4] - xs[10] * xs[5] - GRAVITY * se.sin(xs[7]),
        xs[9] * xs[5] - xs[11] * xs[3] + GRAVITY * se.cos(xs[7]) * se.sin(xs[6]),
        xs[10] * xs[3] - xs[9] * xs[4] + GRAVITY * se.cos(xs[7]) * se.cos(xs[6]),
        xs[9] + xs[10] * se.sin(xs[6]) * se.tan(xs[7]) + xs[11] * se.cos(xs[6]) * se.tan(xs[7]),
        xs[10] * se.cos(xs[6]) - xs[11] * se.sin(xs[6]),
        xs[10] * se.sin(xs[6]) / se.cos(xs[7]) + xs[11] * se.cos(xs[6]) / se.cos(xs[7]),
        xs[10] * xs[11] * ((JY - JZ) / JX),
        xs[9] * xs[11] * ((JZ - JX) / JY),
        xs[9] * xs[10] * ((JX - JY) / JZ),
    ]
)
f_residual_symbolic = se.DenseMatrix(
    [
        0,
        0,
        0,
        1 / 100 * (xs[1] ** 2 - xs[7]),
        1 / 200 * (xs[0] ** 2 + (xs[2] - 2) ** 2),
        1 / 10 * (xs[1] - xs[2] - xs[4] ** 2),
        0,
        0,
        0,
        xs[10] * xs[11] * ((JY0 - JZ0) / JX0 - (JY - JZ) / JX),
        xs[9] * xs[11] * ((JZ0 - JX0) / JY0 - (JZ - JX) / JY),
        xs[9] * xs[10] * ((JX0 - JY0) / JZ0 - (JX - JY) / JZ),
    ]
)
g_symbolic = se.DenseMatrix(
    [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [-1 / MASS, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1 / JX, 0, 0],
        [0, 0, 1 / JY, 0],
        [0, 0, 0, 1 / JZ],
    ]
)
g_residual_symbolic = se.DenseMatrix(
    [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [-(1 / MASS0 - 1 / MASS), 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, (1 / JX0 - 1 / JX), 0, 0],
        [0, 0, (1 / JY0 - 1 / JY), 0],
        [0, 0, 0, (1 / JZ0 - 1 / JZ)],
    ]
)

# Need to be fixed
s_symbolic_deterministic = se.Matrix([[0 for i in range(5)] for j in range(5)])
s_symbolic_stochastic = (
    0.25
    * dt
    * se.Matrix(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ]
    )
)

# Callable Functions
f = dyn_wrapper(f_symbolic, xs)
g = dyn_wrapper(g_symbolic, xs)
sigma_deterministic = dyn_wrapper(s_symbolic_deterministic, xs)
sigma_stochastic = dyn_wrapper(s_symbolic_stochastic, xs)
f_residual = dyn_wrapper(f_residual_symbolic, xs)
g_residual = dyn_wrapper(g_residual_symbolic, xs)

# System Dynamics
def actual_f(z: NDArray) -> NDArray:
    """Synthesizes modeled and unmodeled (residual) uncontrolled dynamics.

    Arguments
        z: state vector

    Returns
        f(z) + f_residual(z)

    """
    return f(z) + f_residual(z)


def actual_g(z: NDArray) -> NDArray:
    """Synthesizes modeled and unmodeled (residual) controlled dynamics.

    Arguments
        z: state vector

    Returns
        g(z) + g_residual(z)

    """
    return g(z) + g_residual(z)


deterministic_dynamics = control_affine_system_deterministic(actual_f, actual_g)
stochastic_dynamics = control_affine_system_stochastic(actual_f, actual_g, sigma_stochastic, dt)

# Step Forward
deterministic_step = first_order_forward_euler(deterministic_dynamics, dt)
stochastic_step = first_order_forward_euler(stochastic_dynamics, dt)

# Determine dimensions
nControls = g(np.zeros((len(xs),))).shape[1]
u0 = np.zeros((nControls,))
