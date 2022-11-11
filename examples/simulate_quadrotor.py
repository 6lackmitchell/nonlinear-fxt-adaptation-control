"""Module docstring??"""
from core.simulate import simulate

VEHICLE = "quadrotor"
LEVEL = "dynamic_6dof"
SITUATION = "wind_field"

ENDTIME = 30.0
TIMESTEP = 1e-2

SUCCESS = simulate(ENDTIME, TIMESTEP, VEHICLE, LEVEL, SITUATION)
