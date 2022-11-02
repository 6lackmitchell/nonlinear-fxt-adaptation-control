"""Module docstring??"""
import matplotlib.pyplot as plt
from core.visualize import visualize

VEHICLE = "sinusoidal"
LEVEL = "uncontrolled"
SITUATION = "standard"

ROOT_DIR = "data"

figure_list = visualize(VEHICLE, LEVEL, SITUATION, ROOT_DIR)

plt.show()
