"""Module docstring??"""
import matplotlib.pyplot as plt
from core.visualize import visualize

VEHICLE = "pendulum"
LEVEL = "forced"
SITUATION = "standard"

ROOT_DIR = "data"

figure_list = visualize(VEHICLE, LEVEL, SITUATION, ROOT_DIR)

plt.show()
