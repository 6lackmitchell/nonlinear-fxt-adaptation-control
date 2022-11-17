"""Module docstring??"""
from core.simulate import simulate

VEHICLE = "quadrotor"
LEVEL = "dynamic_6dof"
SITUATION = "wind_field"

ENDTIME = 20.0
TIMESTEP = 1e-3

SUCCESS = simulate(ENDTIME, TIMESTEP, VEHICLE, LEVEL, SITUATION)
