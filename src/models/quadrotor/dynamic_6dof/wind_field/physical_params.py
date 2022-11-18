"""physical_params.py

Defines the physical parameters involved in simulating the quadrotor
model in the arbitrary_disturbance situation.

Required Parameters:
    GRAVITY
    MASS0
    JX0
    JY0
    JZ0
    MASS
    JX
    JY
    JZ
    U_MAX

"""

import numpy as np
import symengine as se

# Acceleration due to gravity
GRAVITY = 9.81  # meters / sec^2

# Actual Vehicle Parameters -- Taken from AscTec Hummingbird
# (https://productz.com/en/asctec-hummingbird/p/xKp1)
ARM_LENGTH = 0.17  # meters
MASS0 = 0.71  # kg
JX0 = 0.00365  # kg m^2
JY0 = 0.00368  # kg m^2
JZ0 = 0.00703  # kg m^2

# Estimated Vehicle Parameters
MASS = 0.75  # kg
JX = 0.01  # kg m^2
JY = 0.01  # kg m^2
JZ = 0.01  # kg m^2
JX = 1  # kg m^2
JY = 1  # kg m^2
JZ = 1  # kg m^2
JX = JX0  # kg m^2
JY = JY0  # kg m^2
JZ = JZ0  # kg m^2

# Control Constraints
K1 = K2 = 1e-1
F_MAX = 4.0 * MASS * GRAVITY  # propeller force control constraint
D_MAX = F_MAX / (4 * K1)
TX_MAX = ARM_LENGTH * (F_MAX / 4.0)
TY_MAX = ARM_LENGTH * (F_MAX / 4.0)
TZ_MAX = 2 * K2 * D_MAX

# Control input constraints
U_MAX = np.array([F_MAX, TX_MAX, TY_MAX, TZ_MAX])
U_MIN = np.array([0, -TX_MAX, -TY_MAX, -TZ_MAX])

# Residual Dynamics
def f_residual_symbolic(xs: list) -> se.DenseMatrix:
    """Returns a symbolic expression in the form of a DenseMatrix for the
    residual drift dynamics in the system.

    Arguments:
        xs: symbolic states

    Returns:
        residual_f: residual drift dynamics

    """
    residual_f = [
        0,
        0,
        0,
        0.0 * (xs[1] ** 2 - xs[7]) * xs[2],
        0.0 * (xs[0] ** 2 + (xs[2] - 2) ** 2) * xs[3],
        0.0 * (xs[1] - xs[2] - xs[4] ** 2) * xs[0] / MASS,
        0,
        0,
        0,
        xs[10] * xs[11] * ((JY0 - JZ0) / JX0 - (JY - JZ) / JX),
        xs[9] * xs[11] * ((JZ0 - JX0) / JY0 - (JZ - JX) / JY),
        xs[9] * xs[10] * ((JX0 - JY0) / JZ0 - (JX - JY) / JZ),
    ]
    residual_f = [
        0,
        0,
        0,
        0.0,
        0.0,
        0.0,
        0,
        0,
        0,
        0,
        0,
        0,
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
    # residual_g = [
    #     [0, 0, 0, 0],
    #     [0, 0, 0, 0],
    #     [0, 0, 0, 0],
    #     [0, 0, 0, 0],
    #     [0, 0, 0, 0],
    #     [-1 / MASS, 0, 0, 0],
    #     [0, 0, 0, 0],
    #     [0, 0, 0, 0],
    #     [0, 0, 0, 0],
    #     [0, 1 / JX, 0, 0],
    #     [0, 0, 1 / JY, 0],
    #     [0, 0, 0, 1 / JZ],
    # ]

    residual_g = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]

    return se.DenseMatrix(residual_g)
