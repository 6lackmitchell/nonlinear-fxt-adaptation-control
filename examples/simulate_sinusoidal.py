"""Module docstring??"""
from core.simulate import simulate

VEHICLE = "sinusoidal"
LEVEL = "uncontrolled"
SITUATION = "standard"

ENDTIME = 10.0
TIMESTEP = 1e-3

SUCCESS = simulate(ENDTIME, TIMESTEP, VEHICLE, LEVEL, SITUATION)
