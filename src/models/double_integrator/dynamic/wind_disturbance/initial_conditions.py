""""
initial_conditions.py

Specifies the initial conditions for the quadrotor model in the
wind_field situation.

"""
import numpy as np

# Define initial states
xi = np.array([0.0])
yi = np.array([0.0])

# Construct initial state vector
z0 = np.array(
    [
        np.array(
            [
                xi[aa],
                yi[aa],
            ]
        )
        for aa in range(len(xi))
    ]
)

# Initial control vector
u0 = np.array([0.0, 0.0])

# Get agent, state, and control dimensions
N_AGENTS, N_STATES = z0.shape
N_CONTROLS = len(u0)
