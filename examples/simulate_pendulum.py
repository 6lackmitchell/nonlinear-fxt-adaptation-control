"""Simulates the pendulum example from Ch 13.6.1.3 (Forced Pendulum) from the
Koopman Operator textbook."""
from core.simulate import simulate

VEHICLE = "pendulum"
LEVEL = "forced"
SITUATION = "standard"

ENDTIME = 10.0
TIMESTEP = 1e-3

SUCCESS = simulate(ENDTIME, TIMESTEP, VEHICLE, LEVEL, SITUATION)
