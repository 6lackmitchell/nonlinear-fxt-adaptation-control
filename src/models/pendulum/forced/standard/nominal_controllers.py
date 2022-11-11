"""nominal_controllers.py

Defines classes representing nominal (i.e. unconstrained, unsafe, etc.)
controllers for the Rossler dynamical model (uncontrolled) for the standard situation.

"""
from typing import Tuple
from nptyping import NDArray
import numpy as np
from core.controllers.controller import Controller


class ZeroController(Controller):
    """This controls a system with no control input, so it is a dummy class."""

    def __init__(self, ego_id: int):
        super().__init__()

        self.ego_id = ego_id
        self.complete = False

    def _compute_control(self, t: float, z: NDArray) -> Tuple[int, str]:
        """No control input in this system -- computes 0.0

        Arguments:
            t: time (in sec)
            z: full state vector

        Returns:
            code: success (1) or error (0, -1, ...) code
            status: more information on success or cause of error

        """
        self.u = 0.0

        return self.u, 1, "Optimal"


class SinusoidController(Controller):
    """Generates a 1D sinusoidal control input."""

    def __init__(self, ego_id: int):
        super().__init__()

        self.ego_id = ego_id
        self.complete = False

    def _compute_control(self, t: float, z: NDArray) -> Tuple[int, str]:
        """No control input in this system -- computes 0.0

        Arguments:
            t: time (in sec)
            z: full state vector

        Returns:
            code: success (1) or error (0, -1, ...) code
            status: more information on success or cause of error

        """
        self.u = np.sin(t)

        return self.u, 1, "Optimal"
