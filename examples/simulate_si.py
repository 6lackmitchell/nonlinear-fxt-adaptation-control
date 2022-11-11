"""Module docstring??"""
from core.simulate import simulate

VEHICLE = "single_integrator"
LEVEL = "kinematic"
SITUATION = "wind_field"

ENDTIME = 10.0
TIMESTEP = 1e-3

SUCCESS = simulate(ENDTIME, TIMESTEP, VEHICLE, LEVEL, SITUATION)
