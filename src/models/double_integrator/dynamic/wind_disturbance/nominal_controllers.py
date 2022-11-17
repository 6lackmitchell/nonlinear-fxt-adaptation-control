"""nominal_controllers.py

Defines classes representing nominal (i.e. unconstrained, unsafe, etc.)
controllers for the quadrotor dynamical model for the wind_field situation.

"""
from typing import Tuple
import numpy as np
from nptyping import NDArray
from core.controllers.controller import Controller


class LemniscateTrackingController(Controller):
    """Object for cascaded tracking control for the 6-DOF quadrotor dynamical
    model.

    Public Methods:
        populate here

    Public Attributes:
        populate here

    """

    CONTROL_GAINS = {
        "tau": 0.1,  # Time constant
        "f_gerono": 0.1,  # Gerono lemniscate frequency (Hz)
        "a_gerono": 3.0,  # Gerono lemniscate gain
    }

    def __init__(self, ego_id: int):
        super().__init__()

        self.ego_id = ego_id
        self.complete = False
        self.residual_dynamics = np.zeros((12,))  #! Lazy coding -- need to fix!

    def _compute_control(self, t: float, z: NDArray) -> Tuple[int, str]:
        """Computes the input using the cascaded tracking controller for the
        6-DOF quadrotor dynamical model.

        Arguments:
            t: time (in sec)
            z: full state vector

        Returns:
            code: success (1) or error (0, -1, ...) code
            status: more information on success or cause of error

        """
        # Compute desired accelerations
        tc = self.CONTROL_GAINS["tau"]
        xdot, ydot = self.get_desired_velocity(t, z[self.ego_id], tc)

        self.u = np.array([xdot, ydot])

        return self.u, 1, "Optimal"

    def get_desired_velocity(self, t: float, ze: NDArray, tc: float) -> NDArray:
        """Computes the desired velocity of the vehicle in the XY
        coordinates in the inertial frame.

        Arguments:
            TBD

        Returns:
            TBD

        """
        # Gerono Lemniscate Trajectory
        B = 2 * np.pi * self.CONTROL_GAINS["f_gerono"]
        x_g = self.CONTROL_GAINS["a_gerono"] * np.sin(B * t)
        y_g = self.CONTROL_GAINS["a_gerono"] * np.sin(B * t) * np.cos(B * t)
        xdot_g = B * self.CONTROL_GAINS["a_gerono"] * np.cos(B * t)
        ydot_g = B * self.CONTROL_GAINS["a_gerono"] * (np.cos(B * t) ** 2 - np.sin(B * t) ** 2)

        # Compute commanded xdot and ydot
        xdot_c = -2 / tc * (ze[0] - x_g) + xdot_g - self.residual_dynamics[0]
        ydot_c = -2 / tc * (ze[1] - y_g) + ydot_g - self.residual_dynamics[1]

        return xdot_c, ydot_c
