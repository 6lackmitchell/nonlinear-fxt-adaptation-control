"""Module docstring??"""
from core.simulate import simulate

VEHICLE = "double_integrator"
LEVEL = "dynamic"
SITUATION = "wind_disturbance"

ENDTIME = 10.0
TIMESTEP = 1e-3

SUCCESS = simulate(ENDTIME, TIMESTEP, VEHICLE, LEVEL, SITUATION)
