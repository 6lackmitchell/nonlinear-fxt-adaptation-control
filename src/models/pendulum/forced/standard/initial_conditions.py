""""
initial_conditions.py

Specifies the initial conditions for the quadrotor model in the
wind_field situation.

"""
import numpy as np

# Define initial states
x1i = np.array([np.random.uniform(high=np.pi / 2, low=-np.pi / 2)])
x2i = np.array([np.random.uniform(high=np.pi / 2, low=-np.pi / 2)])

# Construct initial state vector
z0 = np.array(
    [
        np.array(
            [
                x1i[aa],
                x2i[aa],
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
