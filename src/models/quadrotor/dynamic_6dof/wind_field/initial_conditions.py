""""
initial_conditions.py

Specifies the initial conditions for the quadrotor model in the
wind_field situation.

"""
import numpy as np

# Define initial states
ei = np.array([0.1])
ni = np.array([0.1])
di = np.array([0.5])
ui = np.array([0.0])
vi = np.array([0.0])
wi = np.array([0.5])
phii = np.array([0.0])
thetai = np.array([0.0])
psii = np.array([0.0])
pi = np.array([0.0])
qi = np.array([0.0])
ri = np.array([0.0])

# Construct initial state vector
z0 = np.array(
    [
        np.array(
            [
                ei[aa],
                ni[aa],
                di[aa],
                ui[aa],
                vi[aa],
                wi[aa],
                phii[aa],
                thetai[aa],
                psii[aa],
                pi[aa],
                qi[aa],
                ri[aa],
            ]
        )
        for aa in range(len(ei))
    ]
)

# Initial control vector
u0 = np.array([0.0, 0.0, 0.0, 0.0])

# Get agent, state, and control dimensions
N_AGENTS, N_STATES = z0.shape
N_CONTROLS = len(u0)
