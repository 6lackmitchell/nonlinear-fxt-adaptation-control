"""Module docstring??"""
import matplotlib.pyplot as plt
from core.visualize import visualize

VEHICLE = "double_integrator"
LEVEL = "dynamic"
SITUATION = "wind_disturbance"

ROOT_DIR = "data"

figure_list = visualize(VEHICLE, LEVEL, SITUATION, ROOT_DIR)

plt.show()
