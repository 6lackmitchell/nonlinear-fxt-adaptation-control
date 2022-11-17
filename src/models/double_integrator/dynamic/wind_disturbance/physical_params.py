"""physical_params.py

Defines the physical parameters involved in simulating the quadrotor
model in the arbitrary_disturbance situation.

Required Parameters:
    U_MAX

"""

import numpy as np
import symengine as se

# Control input constraints
VX_MAX = VY_MAX = 100.0
U_MAX = np.array([VX_MAX, VY_MAX])

# Residual Dynamics
def f_residual_symbolic(xs: list) -> se.DenseMatrix:
    """Returns a symbolic expression in the form of a DenseMatrix for the
    residual drift dynamics in the system.

    Arguments:
        xs: symbolic states

    Returns:
        residual_f: residual drift dynamics

    """
    # residual_f = [xs[0] ** 2 * xs[1], xs[0] * xs[1] ** 2]
    scale = 1.0
    gain = 5.0
    residual_f = [
        gain * (se.sin((xs[0] ** 2) / scale) + se.cos((xs[0] * xs[1]) / scale)),
        gain * (se.cos((xs[1]) / scale) ** 2),
    ]
    return se.DenseMatrix(residual_f)


def g_residual_symbolic(xs: list) -> se.DenseMatrix:
    """Returns a symbolic expression in the form of a DenseMatrix for the
    residual matched dynamics in the system.

    Arguments:
        xs: symbolic states

    Returns:
        residual_g: residual matched dynamics

    """
    residual_g = [[0, 0], [0, 0]]

    return se.DenseMatrix(residual_g)
