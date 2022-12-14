"""nominal_controllers.py

Defines classes representing nominal (i.e. unconstrained, unsafe, etc.)
controllers for the quadrotor dynamical model for the arbitrary_disturbance situation.

"""
from typing import Tuple
import numpy as np
from nptyping import NDArray
from core.controllers.controller import Controller
from .physical_params import GRAVITY, MASS, JX, JY, JZ, U_MAX
from ..rotations import rotation_body_frame_to_inertial_frame

# from .initial_conditions import *


class CascadedTrackingController(Controller):
    """Object for cascaded tracking control for the 6-DOF quadrotor dynamical
    model.

    Public Methods:
        populate here

    Public Attributes:
        populate here

    """

    CONTROL_GAINS = {
        "k_r": 1.0,  # Yaw rate proportional gain
        "k_phi": 125.0,  # Roll moment proportional gain
        "k_theta": 125.0,  # Pitch moment proportional gain
        "k_psi": 1.0,  # Yaw moment proportional gain
        "zeta": 1.0,  # Damping ratio
        "tau_f": 0.5,  # Time constant (force control)
        "tau_m": 0.2,  # Time constant (moment control)
        "rate": 2.0,  # Exponential decay rate
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
        ze = z[self.ego_id]

        # Get rotation matrix
        rotation = rotation_body_frame_to_inertial_frame(ze)

        # Compute desired accelerations
        dr = self.CONTROL_GAINS["zeta"]
        tc = self.CONTROL_GAINS["tau_f"]
        xddot, yddot, zddot = self.get_desired_acceleration(t, ze, dr, tc, rotation)

        # Compute force output
        force = self.compute_force_command(zddot, rotation)

        # Compute momentts
        pitch_moment, roll_moment, yaw_moment = self.compute_moment_commands(
            t, ze, force, xddot, yddot, rotation
        )

        self.u = np.array([force, pitch_moment, roll_moment, yaw_moment])

        return self.u, 1, "Optimal"

    def compute_force_command(self, zddot: float, rotation: NDArray) -> float:
        """Computes the force control for the quadrotor tracking controller. The
        force control is the first level of the cascaded controller, and the
        moments depend on the computed force value.

        Arguments:
            zddot: desired acceleration (in m/s^2) in the positive z direction
            rotation: body to inertial frame rotation matrix

        Returns:
            force: control input in N

        """
        force = MASS * np.max([0, (GRAVITY + zddot) / -rotation[2, 2]])

        return np.clip(force, 0, U_MAX[0])

    def compute_moment_commands(
        self, t: float, ze: NDArray, force: float, xddot: float, yddot: float, rotation: NDArray
    ) -> Tuple[float, float, float]:
        """Computes the force control for the quadrotor tracking controller. The
        force control is the first level of the cascaded controller, and the
        moments depend on the computed force value.

        Arguments:
            t: time (in sec)
            ze: ego vehicle state vector
            force: thrust force (in N) computed by first level of cascaded controller
            xddot: desired acceleration (in m/s^2) in the inertial x direction
            yddot: desired acceleration (in m/s^2) in the inertial y direction
            rotation: rotation matrix from body-fixed frame to inertial frame

        Returns:
            pitch_moment
            roll_moment
            yaw_moment

        """
        pitch_moment = 0.0
        roll_moment = 0.0
        yaw_moment = 0.0

        theta = 0.0  ## NEED TO FIX: this is not correct

        if force > 0.0:
            R13_c = -MASS * xddot / force
            R23_c = -MASS * yddot / force
            R33_c = theta + np.pi / 2

            tc_R = 0.5 * self.CONTROL_GAINS["tau_f"]
            if t > 1.0:
                tc_Rf = 0.1 * self.CONTROL_GAINS["tau_f"]
                tc_R = tc_Rf + (self.CONTROL_GAINS["tau_m"] - tc_Rf) * np.exp(
                    -self.CONTROL_GAINS["rate"] * (t - 1.0)
                )

            R13dot_c = -(rotation[0, 2] - R13_c) / (tc_R)
            R23dot_c = -(rotation[1, 2] - R23_c) / (tc_R)

            p_c = -R13dot_c * rotation[0, 1] - R23dot_c * rotation[1, 1] - self.residual_dynamics[9]
            q_c = R13dot_c * rotation[0, 0] + R23dot_c * rotation[1, 2] - self.residual_dynamics[10]
            r_c = 0 - self.residual_dynamics[11]  # -k_r * (ze[8] - theta + np.pi/2)

            pitch_moment = (
                -self.CONTROL_GAINS["k_phi"] * (ze[9] - p_c) * JX - (JY - JZ) * ze[10] * ze[11]
            )
            roll_moment = (
                -self.CONTROL_GAINS["k_theta"] * (ze[10] - q_c) * JY - (JZ - JX) * ze[9] * ze[11]
            )
            yaw_moment = 0  # -k_psi * (ze[11] - r_c) * Jz - (Jx - Jy)*ze[9]*ze[10]

        return (
            np.clip(pitch_moment, -U_MAX[1], U_MAX[1]),
            np.clip(roll_moment, -U_MAX[2], U_MAX[2]),
            np.clip(yaw_moment, -U_MAX[3], U_MAX[3]),
        )

    def get_desired_acceleration(
        self, t: float, ze: NDArray, dr: float, tc: float, rotation: NDArray
    ) -> NDArray:
        """Computes the desired acceleration of the quadrotor in the XYZ
        coordinates in the inertial frame.

        Arguments:
            TBD

        Returns:
            TBD

        """
        # Time Constant Adjustments
        tc_x = tc
        tc_y = tc
        tc_z = tc

        # Compute velocities in inertial frame
        xdot, ydot, zdot = (
            rotation @ ze[3:6] + self.residual_dynamics[0:3]
        )  # + regressor(state)[0:3] @ thetaHat

        # Setpoint -- take off
        x_c = ze[0]
        y_c = ze[1]
        xdot_c = 0.0
        ydot_c = 0.0
        xddot_c = 0.0
        yddot_c = 0.0
        z_c = 2.0
        zdot_c = 0.0
        zddot_c = 0.0

        # Gerono Lemniscate Trajectory -- override takeoff when altitude sufficiently high
        if ze[2] > 1.0:
            B = 2 * np.pi * self.CONTROL_GAINS["f_gerono"]

            x_c = self.CONTROL_GAINS["a_gerono"] * np.sin(B * t)
            y_c = self.CONTROL_GAINS["a_gerono"] * np.sin(B * t) * np.cos(B * t)

            xdot_c = B * self.CONTROL_GAINS["a_gerono"] * np.cos(B * t)
            ydot_c = B * self.CONTROL_GAINS["a_gerono"] * (np.cos(B * t) ** 2 - np.sin(B * t) ** 2)

            xddot_c = -(B**2) * self.CONTROL_GAINS["a_gerono"] * np.sin(B * t)
            yddot_c = -4 * B**2 * self.CONTROL_GAINS["a_gerono"] * np.sin(B * t) * np.cos(B * t)

        xddot = (-2 * dr / tc_x * (xdot - xdot_c) - (ze[0] - x_c) / tc_x**2 + xddot_c) - rotation[
            0
        ] @ self.residual_dynamics[
            3:6
        ]  # - rotation[0] @ regressor(state)[3:6] @ thetaHat
        yddot = (-2 * dr / tc_y * (ydot - ydot_c) - (ze[1] - y_c) / tc_y**2 + yddot_c) - rotation[
            1
        ] @ self.residual_dynamics[
            3:6
        ]  # - rotation[1] @ regressor(state)[3:6] @ thetaHat
        zddot = (-2 * dr / tc_z * (zdot - zdot_c) - (ze[2] - z_c) / tc_z**2 + zddot_c) - rotation[
            2
        ] @ self.residual_dynamics[
            3:6
        ]  # - rotation[2] @ regressor(state)[3:6] @ thetaHat

        return xddot, yddot, zddot
