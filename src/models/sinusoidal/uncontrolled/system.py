"""system.py

Defines functions describing the system dynamics for the Rossler system.

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
xs = se.symbols(["x1", "x2"])

# Define symbolic system dynamics
f_symbolic = se.DenseMatrix([1, -1])
f_residual_symbolic = se.DenseMatrix([se.sin(xs[1]), se.cos(xs[0])])
g_symbolic = se.DenseMatrix([[0], [1]])
g_residual_symbolic = se.DenseMatrix([[xs[1]], [-xs[1]]])

# Need to be fixed
s_symbolic_deterministic = se.Matrix([[0 for i in range(2)] for j in range(2)])
s_symbolic_stochastic = (
    0.25
    * dt
    * se.Matrix(
        [
            [1, 0],
            [0, 1],
        ]
    )
)

# Callable Functions
f = dyn_wrapper(f_symbolic, xs)
g_wrapped = dyn_wrapper(g_symbolic, xs)
g = lambda x: g_wrapped(x)[:, np.newaxis]

sigma_deterministic = dyn_wrapper(s_symbolic_deterministic, xs)
sigma_stochastic = dyn_wrapper(s_symbolic_stochastic, xs)
f_residual = dyn_wrapper(f_residual_symbolic, xs)
g_residual_wrapped = dyn_wrapper(g_residual_symbolic, xs)
g_residual = lambda x: g_residual_wrapped(x)[:, np.newaxis]

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
