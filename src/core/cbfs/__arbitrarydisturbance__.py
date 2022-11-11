"""__windfield__.py

Initialization of the cbf module when using the quadrotor/dynamic_6dof situation
wind_field.

"""
from typing import Callable
import numpy as np
from .cbf import Cbf

from .symbolic_cbfs.obstacle_avoidance import (
    h_oa1,
    dhdx_oa1,
    d2hdx2_oa1,
    h_oa2,
    dhdx_oa2,
    d2hdx2_oa2,
)


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
cbfs_individual = [
    Cbf(h_oa1, dhdx_oa1, d2hdx2_oa1, linear_class_k(K_DEFAULT), h_oa1),
    Cbf(h_oa2, dhdx_oa2, d2hdx2_oa2, linear_class_k(K_DEFAULT), h_oa2),
]

# cbfs_individual = []
cbfs_pairwise = []

cbf0 = np.zeros((len(cbfs_individual) + len(cbfs_pairwise),))
