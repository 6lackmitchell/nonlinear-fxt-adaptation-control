"""replay.py

Contains the interface to the replay function, which plays back trajectories
of various simulated quantities in the quadrotor/dynamic_6dof/wind_field
scenario.

Methods:
    replay

"""
import os
import glob
import pickle
from typing import List, Optional
from nptyping import NDArray
import numpy as np

import matplotlib
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# from core.visualizing.helpers import get_circle, get_ex

from .initial_conditions import N_AGENTS
from .timing_params import dt
from ..system import f


matplotlib.use("Qt5Agg")


import matplotlib.pyplot as plt

# from matplotlib import animation


matplotlib.rcParams.update({"figure.autolayout": True})

N = 4 * N_AGENTS
plt.style.use(["ggplot"])
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, N)))


def set_edges_black(ax: matplotlib.axes.Axes) -> None:
    """Method called to set the edges of the specified plot to black.

    Arguments:
        ax: matplotlib.axes.Axes object

    Returns:
        None

    """
    ax.spines["bottom"].set_color("#000000")
    ax.spines["top"].set_color("#000000")
    ax.spines["right"].set_color("#000000")
    ax.spines["left"].set_color("#000000")


# Define Visualization Constants
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
COLORS[0] = COLORS[1]
COLORS.reverse()
LINE_WIDTH = 2
DASH = [3, 2]
COLOR_IDX = np.array(range(0, 2 * N_AGENTS)).reshape(N_AGENTS, 2)


def replay(filepath: str, fname: Optional[str] = None) -> List[matplotlib.figure.Figure]:
    """Replays the trajectory data from various simulated quantities in the
    quadrotor/dynamic_6dof/wind_field situation.

    Arguments:
        filepath: path to the directory of the saved data file

    Returns:
        state_figs: list to handles of figure objects

    """
    # Get the file
    if fname is None:
        ftype = r"*.pkl"
        files = glob.glob(filepath + ftype)
        files.sort(key=os.path.getmtime)
        filename = files[-1]
    else:
        filename = filepath + fname

    print(filename)

    # Load data
    with open(filename, "rb") as f:
        data = pickle.load(f)

        x = np.array([data[a]["x"] for a in data.keys()])
        u = np.array([data[a]["u"] for a in data.keys()])
        k = np.array([data[a]["kgains"] if a < 3 else None for a in data.keys()][0:3])
        u0 = np.array([data[a]["u0"] for a in data.keys()])
        f_err = np.array([data[a]["f_error"] for a in data.keys()])
        f_est = np.array([data[a]["f_est"] for a in data.keys()])
        ii = int(data[0]["ii"] / dt) - 1

    # Load baseline data -- no disturbance
    with open(filepath + "nominal_controller_no_disturbance.pkl", "rb") as f:
        data = pickle.load(f)
        xr = np.array([data[a]["x"] for a in data.keys()])

    # Load baseline data -- no estimation
    with open(filepath + "nominal_controller_no_estimation.pkl", "rb") as f:
        data = pickle.load(f)
        xb = np.array([data[a]["x"] for a in data.keys()])

    # Compute derived quantities
    tf = ii * dt
    t = np.linspace(dt, tf, int(tf / dt))
    # t = t[:-1000]

    state_figs = generate_state_figures(t, x, xr, xb)
    estimation_figs = generate_estimation_figures(t, f_est, f_err)

    # Generate Control Figs
    control_figs = generate_control_figures(t, u, u0)

    # Generate CBF Figs
    cbf_figs = generate_cbf_figures(t, x)

    return state_figs + estimation_figs + control_figs + cbf_figs


def generate_state_figures(t: NDArray, x: NDArray, xr: NDArray, xb: NDArray) -> List:
    """Generates static figures for plotting the state trajectories.

    Arguments:
        t: time vector
        x: time history of the state vector
        xr: time history of reference (no disturbance) state trajectory
        xb: time history of baseline (no estimation) state trajectory

    Returns:
        figs: list to handles of figure objects

    """
    # Set up figure
    fig = plt.figure(figsize=(10, 10))
    ax_x = fig.add_subplot(311)
    ax_y = fig.add_subplot(312)
    ax_z = fig.add_subplot(313)
    set_edges_black(ax_x)
    set_edges_black(ax_y)
    set_edges_black(ax_z)
    ax_x.plot(t, x[0, : len(t), 0], label="x")
    ax_y.plot(t, x[0, : len(t), 1], label="y")
    ax_z.plot(t, x[0, : len(t), 2], label="z")
    ax_x.legend()
    ax_y.legend()
    ax_z.legend()

    # Attitude figure
    fig2 = plt.figure(figsize=(10, 10))
    ax_phi = fig2.add_subplot(311)
    ax_the = fig2.add_subplot(312)
    ax_psi = fig2.add_subplot(313)
    set_edges_black(ax_phi)
    set_edges_black(ax_the)
    set_edges_black(ax_psi)
    ax_phi.plot(t, x[0, : len(t), 6], label=r"$\phi$")
    ax_the.plot(t, x[0, : len(t), 7], label=r"$\theta$")
    ax_psi.plot(t, x[0, : len(t), 8], label=r"$\psi$")
    ax_phi.legend()
    ax_the.legend()
    ax_psi.legend()

    # Set up figure
    fig_xy = plt.figure(figsize=(10, 10))
    ax_xy = fig_xy.add_subplot(111)

    # Add Obstacles
    ax_xy.add_artist(
        Circle(
            (-2.0, 0.0),
            1.0,
            edgecolor="black",
            facecolor=(0, 0, 0),
        )
    )
    ax_xy.add_artist(
        Circle(
            (1.25, -1.25),
            1.0,
            edgecolor="black",
            facecolor=(0, 0, 0),
        )
    )
    set_edges_black(ax_xy)
    ax_xy.plot(x[0, : len(t), 0], x[0, : len(t), 1], linewidth=3, label="FxT Estimation Controller")
    ax_xy.plot(
        xr[0, : len(t), 0], xr[0, : len(t), 1], ".-", linewidth=3, label="Reference: No Disturbance"
    )
    ax_xy.plot(
        xb[0, : len(t), 0], xb[0, : len(t), 1], ":", linewidth=3, label="Baseline: No Estimation"
    )
    ax_xy.legend(fancybox=True)
    ax_xy.set_xlim([-10, 10])
    ax_xy.set_ylim([-10, 10])

    # Figure list
    figs = [fig, fig2, fig_xy]

    return figs


def generate_estimation_figures(t: NDArray, f_est: NDArray, f_err: NDArray) -> List:
    """Generates static figures for plotting the state trajectories.

    Arguments:
        t: time vector
        f_err: time history of the unknown function estimation error

    Returns:
        figs: list to handles of figure objects

    """

    # Set up figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    set_edges_black(ax)
    ax.plot(t, f_est[0, : len(t), 9])
    ax.plot(t, f_est[0, : len(t), 10])
    ax.plot(t, f_est[0, : len(t), 11])
    ax.plot(t, f_est[0, : len(t), 9] + f_err[0, : len(t), 9], ":")
    ax.plot(t, f_est[0, : len(t), 10] + f_err[0, : len(t), 10], ":")
    ax.plot(t, f_est[0, : len(t), 11] + f_err[0, : len(t), 11], ":")
    # ax.set_ylim([-10, 10])

    # Figure list
    figs = [fig]

    return figs


def generate_control_figures(
    t: NDArray,
    u: NDArray,
    u0: NDArray,
) -> List:
    """Generates static figures for plotting the state trajectories.

    Arguments:
        t: time vector
        f_err: time history of the unknown function estimation error

    Returns:
        figs: list to handles of figure objects

    """
    # Set up figure
    lwidth = 5
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(411)
    ax2 = fig.add_subplot(412)
    ax3 = fig.add_subplot(413)
    ax4 = fig.add_subplot(414)
    set_edges_black(ax1)
    set_edges_black(ax2)
    set_edges_black(ax3)
    set_edges_black(ax4)

    # Control Plots
    ax1.plot(t, u[0, : len(t), 0], label=r"$F$ (Test)", linewidth=lwidth)
    ax2.plot(t, u[0, : len(t), 1], label=r"$\tau_{\phi}$ (Test)", linewidth=lwidth)
    ax3.plot(t, u[0, : len(t), 2], label=r"$\tau_{\theta}$ (Test)", linewidth=lwidth)
    ax4.plot(t, u[0, : len(t), 3], label=r"$\tau_{\psi}$ (Test)", linewidth=lwidth)
    ax1.plot(t, u0[0, : len(t), 0], ":", label=r"$F$ (Test)", linewidth=lwidth)
    ax2.plot(t, u0[0, : len(t), 1], ":", label=r"$\tau_{\phi}$ (Test)", linewidth=lwidth)
    ax3.plot(t, u0[0, : len(t), 2], ":", label=r"$\tau_{\theta}$ (Test)", linewidth=lwidth)
    ax4.plot(t, u0[0, : len(t), 3], ":", label=r"$\tau_{\psi}$ (Test)", linewidth=lwidth)
    # ax1.plot(t, u_nai[0, : len(t), 0], label=r"$v_x$ (Naive CBF)", linewidth=lwidth)
    # ax1.plot(t, u_rob[0, : len(t), 0], "-.", label=r"$v_x$ (Robust CBF)", linewidth=lwidth)
    # ax1.plot(t, u_rad[0, : len(t), 0], ":", label=r"$v_x$ (Robust-Adaptive CBF)", linewidth=lwidth)
    # ax2.plot(t, u_nai[0, : len(t), 1], label=r"$v_y$ (Naive CBF)", linewidth=lwidth)
    # ax2.plot(t, u_rob[0, : len(t), 1], "-.", label=r"$v_y$ (Robust CBF)", linewidth=lwidth)
    # ax2.plot(t, u_rad[0, : len(t), 1], ":", label=r"$v_y$ (Robust-Adaptive CBF)", linewidth=lwidth)

    # ax1.set_xlim([-0.1, 10])
    # ax2.set_xlim([-0.1, 10])
    # ax1.set_ylim([-20, 35])
    # ax2.set_ylim([-20, 35])
    ax1.legend(fancybox=True, fontsize=20, loc="upper right")
    ax2.legend(fancybox=True, fontsize=20, loc="upper right")
    ax3.legend(fancybox=True, fontsize=20, loc="upper right")
    ax4.legend(fancybox=True, fontsize=20, loc="upper right")
    # ax1.set(ylabel=r"$v_x$ (m\sec)")
    # ax1.set(ylabel=r"$v_x$ (m\sec)")
    # ax1.set(ylabel=r"$v_x$ (m\sec)")
    ax4.set(xlabel="Time (sec)")

    # # Font Sizes
    # for item in (
    #     [ax1.title, ax1.xaxis.label, ax1.yaxis.label]
    #     + ax1.get_xticklabels()
    #     + ax1.get_yticklabels()
    # ):
    #     item.set_fontsize(35)
    # for item in (
    #     [ax2.title, ax2.xaxis.label, ax2.yaxis.label]
    #     + ax2.get_xticklabels()
    #     + ax2.get_yticklabels()
    # ):
    #     item.set_fontsize(35)

    # Figure list
    figs = [fig]

    return figs


def generate_cbf_figures(t: NDArray, x: List[NDArray]) -> List:
    """Generates static figures for plotting the state trajectories.

    Arguments:
        t: time vector
        f_err: time history of the unknown function estimation error

    Returns:
        figs: list to handles of figure objects

    """
    # Set up figure
    lwidth = 5
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(111)
    set_edges_black(ax1)

    # Generate CBF Values
    cx1 = -2.5
    cy1 = 0.0
    cx2 = 1.25
    cy2 = -1.25

    # Derivatives for CBFs
    zdot = np.array([f(x[0, ii])[2] for ii in range(len(t))])
    phidot = np.array([f(x[0, ii])[6] for ii in range(len(t))])
    thedot = np.array([f(x[0, ii])[7] for ii in range(len(t))])

    h_1 = (x[0, : len(t), 0] - cx1) ** 2 + (x[0, : len(t), 1] - cy1) ** 2 - 1
    h_2 = (x[0, : len(t), 0] - cx2) ** 2 + (x[0, : len(t), 1] - cy2) ** 2 - 1
    h_3 = x[0, : len(t), 2] + zdot
    h_4 = 100 * (
        -phidot * np.sin(x[0, : len(t), 6]) * np.cos(x[0, : len(t), 7])
        - thedot * np.cos(x[0, : len(t), 6]) * np.sin(x[0, : len(t), 7])
        + (np.cos(x[0, : len(t), 6]) * np.cos(x[0, : len(t), 7]) - np.cos(np.pi / 2)) * 0.1
    )
    h_4_0 = np.cos(x[0, : len(t), 6]) * np.cos(x[0, : len(t), 7]) - np.cos(np.pi / 2)

    # CBF Plots
    ax1.plot(t, np.zeros((len(t),)), "k", label="Boundary", linewidth=lwidth)
    ax1.plot(t, h_1, label=r"$h_1$ (Test CBF)", linewidth=lwidth)
    ax1.plot(t, h_2, label=r"$h_2$ (Test CBF)", linewidth=lwidth)
    ax1.plot(t, h_3, label=r"$h_3$ (Test CBF)", linewidth=lwidth)
    ax1.plot(t, h_4_0, label=r"$h_4$ (Test CBF)", linewidth=lwidth)

    # # Inset Axis 1
    # ax_inset1 = inset_axes(
    #     ax1,
    #     width="100%",
    #     height="100%",
    #     bbox_to_anchor=(0.7, 0.05, 0.25, 0.2),
    #     bbox_transform=ax1.transAxes,
    #     loc=3,
    # )
    # set_edges_black(ax_inset1)
    # ax_inset1.plot(t, np.zeros((len(t),)), "k", label="Boundary", linewidth=lwidth)
    # ax_inset1.plot(t, h_1, label=r"$h_1$ (Naive CBF)", linewidth=lwidth)
    # ax_inset1.plot(t, h_2, label=r"$h_2$ (Naive CBF)", linewidth=lwidth)
    # ax_inset1.set_xlim([7.5, 8.5])
    # ax_inset1.set_ylim([-0.2, 0.2])
    # ax_inset1.set(xticklabels=[], yticklabels=[])
    # mark_inset(ax1, ax_inset1, loc1=4, loc2=2, fc="none", ec="0.2", lw=1.5)

    ax1.set_xlim([-0.1, 15])
    # ax1.set_ylim([-0.1, 25])
    ax1.legend(fancybox=True, fontsize=27, loc="upper right")
    ax1.set(xlabel="Time (sec)", ylabel=r"CBF Values")

    # Font Sizes
    for item in (
        [ax1.title, ax1.xaxis.label, ax1.yaxis.label]
        + ax1.get_xticklabels()
        + ax1.get_yticklabels()
    ):
        item.set_fontsize(35)

    # Figure list
    figs = [fig]

    return figs
