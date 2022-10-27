"""rotations.py

Defines functions related to the rotation matrices between quadrotor body-fixed
and inertial frames.

"""
import numpy as np
from nptyping import NDArray


def rotation_body_frame_to_inertial_frame(xs: NDArray) -> NDArray:
    """Computes the rotation matrix from the body-fixed frame to the inertial
    frame based on the current state.

    Arguments:
        xs: current quadrotor state

    Returns:
        rotation_matrix: rotation matrix from body-fixed to inertial frame

    """
    rotation = np.array(
        [
            [
                np.cos(xs[7]) * np.cos(xs[8]),
                np.sin(xs[6]) * np.sin(xs[7]) * np.cos(xs[8]) - np.cos(xs[6]) * np.sin(xs[8]),
                np.cos(xs[6]) * np.sin(xs[7]) * np.cos(xs[8]) + np.sin(xs[6]) * np.sin(xs[8]),
            ],
            [
                np.cos(xs[7]) * np.sin(xs[8]),
                np.sin(xs[6]) * np.sin(xs[7]) * np.sin(xs[8]) + np.cos(xs[6]) * np.cos(xs[8]),
                np.cos(xs[6]) * np.sin(xs[7]) * np.sin(xs[8]) - np.sin(xs[6]) * np.cos(xs[8]),
            ],
            [np.sin(xs[7]), -np.sin(xs[6]) * np.cos(xs[7]), -np.cos(xs[6]) * np.cos(xs[7])],
        ]
    )

    return rotation


def rotation_inertial_frame_to_body_frame(xs: NDArray) -> NDArray:
    """Computes the rotation matrix from the inertial frame to the body-fixed
    frame based on the current state.

    Arguments:
        xs: current quadrotor state

    Returns:
        rotation_matrix: rotation matrix from inertial to body-fixed frame

    """
    rotation = rotation_body_frame_to_inertial_frame(xs).T

    return rotation
