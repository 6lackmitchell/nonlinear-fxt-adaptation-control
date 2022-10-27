"""Module docstring??"""
from core.simulate import simulate

VEHICLE = "quadrotor"
LEVEL = "dynamic_6dof"
SITUATION = "wind_field"

ENDTIME = 40.0
TIMESTEP = 0.01

SUCCESS = simulate(ENDTIME, TIMESTEP, VEHICLE, LEVEL, SITUATION)
