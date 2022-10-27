"""Module docstring??"""
import matplotlib.pyplot as plt
from core.visualize import visualize

VEHICLE = "quadrotor"
LEVEL = "dynamic_6dof"
SITUATION = "wind_field"

ROOT_DIR = "data"

figure_list = visualize(VEHICLE, LEVEL, SITUATION, ROOT_DIR)

plt.show()
