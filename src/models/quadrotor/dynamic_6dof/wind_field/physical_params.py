"""physical_params.py

Defines the physical parameters involved in simulating the quadrotor
model in the wind_field situation.

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
JX = 0.004  # kg m^2
JY = 0.004  # kg m^2
JZ = 0.004  # kg m^2

# Control Constraints
K1 = K2 = 1e-1
F_MAX = 4.0 * MASS * GRAVITY  # propeller force control constraint
D_MAX = F_MAX / (4 * K1)
TX_MAX = ARM_LENGTH * (F_MAX / 4.0)
TY_MAX = ARM_LENGTH * (F_MAX / 4.0)
TZ_MAX = 2 * K2 * D_MAX

# Control input constraints
U_MAX = np.array([F_MAX, TX_MAX, TY_MAX, TZ_MAX])
