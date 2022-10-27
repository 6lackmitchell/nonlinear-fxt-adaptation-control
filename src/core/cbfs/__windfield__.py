"""__windfield__.py

Initialization of the cbf module when using the quadrotor/dynamic_6dof situation
wind_field.

"""
from typing import Callable
import numpy as np
from .cbf import Cbf
from .symbolic_cbfs.collision_avoidance_2d import h_ca, h_rpca, dhdx_rpca, d2hdx2_rpca

# from .symbolic_cbfs.warehouse_safety import h0_road, h_road, dhdx_road, d2hdx2_road
# from .symbolic_cbfs.speed_safety import h_speed, dhdx_speed, d2hdx2_speed


def linear_class_k(k: float) -> Callable:
    """Returns a linear class K function with gain k.

    Arguments:
        k: linear class K function gain

    Returns:
        alpha(h) = k * h
    """

    def alpha(h: float) -> float:
        """Computes the value of the linear class K function k * h for the
        argument h.

        Arguments:
            h: value of the argument (the cbf in most cases)

        Returns:
            k * h: value of the linear class K function evaluated at h

        """
        return k * h

    return alpha


# Define linear class k weights
K_DEFAULT = 1.0
K_COLLISION = 1.0

# Define cbf lists
# cbfs_individual = [
#     Cbf(h_road, dhdx_road, d2hdx2_road, linear_class_k(K_DEFAULT), h0_road),
#     Cbf(h_speed, dhdx_speed, d2hdx2_speed, linear_class_k(K_COLLISION), h_speed),
# ]
# cbfs_pairwise = [
#     Cbf(
#         h_rpca, dhdx_rpca, d2hdx2_rpca, linear_class_k(K_COLLISION), h_ca
#     ),  # Collision Avoidance (ca)
# ]  # RV-CBF

cbfs_individual = []
cbfs_pairwise = []

cbf0 = np.zeros((len(cbfs_individual) + len(cbfs_pairwise),))
