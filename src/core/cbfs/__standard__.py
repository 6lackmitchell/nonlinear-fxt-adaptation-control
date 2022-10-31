"""__windfield__.py

Initialization of the cbf module when using the quadrotor/dynamic_6dof situation
wind_field.

"""
import numpy as np
from .cbf import Cbf

cbfs_individual = []
cbfs_pairwise = []

cbf0 = np.zeros((len(cbfs_individual) + len(cbfs_pairwise),))
