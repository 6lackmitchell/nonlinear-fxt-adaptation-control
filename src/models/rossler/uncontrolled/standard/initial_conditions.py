""""
initial_conditions.py

Specifies the initial conditions for the quadrotor model in the
wind_field situation.

"""
import numpy as np

# Define initial states
x1i = np.array([2.7])
x2i = np.array([8.1])
x3i = np.array([3.2])

# Construct initial state vector
z0 = np.array(
    [
        np.array(
            [
                x1i[aa],
                x2i[aa],
                x3i[aa],
            ]
        )
        for aa in range(len(x1i))
    ]
)

# Initial control vector
u0 = np.array([0.0])

# Get agent, state, and control dimensions
N_AGENTS, N_STATES = z0.shape
N_CONTROLS = len(u0)
