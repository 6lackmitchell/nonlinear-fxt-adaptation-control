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

from core.visualizing.helpers import get_circle

from .initial_conditions import N_AGENTS
from .timing_params import dt


matplotlib.use("Qt5Agg")


import matplotlib.pyplot as plt

# from matplotlib import animation


matplotlib.rcParams.update({"figure.autolayout": True})

N = 5 * N_AGENTS
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

    # List filenames
    fname = [
        "nominal_no_estimation.pkl",
        "noise_free/naive_cbf_nominal_estimation.pkl",
        "noisy/robust_cbf.pkl",
        "noisy/robustadaptive_cbf.pkl",
        "reference.pkl",
    ]

    # Nominal -- no safety no estimation
    with open(filepath + fname[0], "rb") as f:
        data = pickle.load(f)

        x_nom = np.array([data[a]["x"] for a in data.keys()])
        ii = int(data[0]["ii"] / dt) - 1

    # Naive CBF -- assume no model error
    with open(filepath + fname[1], "rb") as f:
        data = pickle.load(f)

        x_nai = np.array([data[a]["x"] for a in data.keys()])
        u_nai = np.array([data[a]["u"] for a in data.keys()])
        h_nai = np.array([data[a]["cbf"] for a in data.keys()])
        f_err_nai = np.array([data[a]["f_error"] for a in data.keys()])
        f_est_nai = np.array([data[a]["f_est"] for a in data.keys()])

    # Robust CBF Condition w/ Estimation
    with open(filepath + fname[2], "rb") as f:
        data = pickle.load(f)

        x_rob = np.array([data[a]["x"] for a in data.keys()])
        u_rob = np.array([data[a]["u"] for a in data.keys()])
        h_rob = np.array([data[a]["cbf"] for a in data.keys()])
        f_err_rob = np.array([data[a]["f_error"] for a in data.keys()])
        f_est_rob = np.array([data[a]["f_est"] for a in data.keys()])

    # Robust-Adaptive CBF Condition w/ Estimation
    with open(filepath + fname[3], "rb") as f:
        data = pickle.load(f)

        x_rad = np.array([data[a]["x"] for a in data.keys()])
        u_rad = np.array([data[a]["u"] for a in data.keys()])
        h_rad = np.array([data[a]["cbf"] for a in data.keys()])
        f_err_rad = np.array([data[a]["f_error"] for a in data.keys()])
        f_est_rad = np.array([data[a]["f_est"] for a in data.keys()])

    # Reference Trajectory
    with open(filepath + fname[4], "rb") as f:
        data = pickle.load(f)

        x_ref = np.array([data[a]["x"] for a in data.keys()])

    # Compute derived quantities
    tf = ii * dt
    t = np.linspace(dt, tf, int(tf / dt))

    # Generate XY Figure
    state_figs = generate_state_figures(t, x_ref, x_nom, x_nai, x_rob, x_rad)

    # Generate Estimation Figure
    f_nai = [f_err_nai, f_est_nai]
    f_rob = [f_err_rob, f_est_rob]
    f_rad = [f_err_rad, f_est_rad]
    estimation_figs = generate_estimation_figures(t, f_nai, f_rob, f_rad)

    # Generate Control Figs
    control_figs = generate_control_figures(t, u_nai, u_rob, u_rad)

    # Generate CBF Figs
    cbf_figs = generate_cbf_figures(t, x_nai, x_rob, x_rad)
    # return cbf_figs

    return state_figs + estimation_figs + control_figs + cbf_figs


def generate_state_figures(
    t: NDArray, xref: NDArray, xnom: NDArray, xnai: NDArray, xrob: NDArray, xrad: NDArray
) -> List:
    """Generates static figures for plotting the state trajectories.

    Arguments:
        t: time vector
        xref: time history of state vector under nominal control input
        xnom: time history of state vector under nominal control input
        xnai: time history of state vector under naive CBF control input
        xrob: time history of state vector under robust CBF control input
        xrad: time history of state vector under robust-adaptive CBF control input

    Returns:
        figs: list to handles of figure objects

    """
    # # Set up figure
    # fig = plt.figure(figsize=(10, 10))
    # ax_x = fig.add_subplot(211)
    # ax_y = fig.add_subplot(212)
    # set_edges_black(ax_x)
    # set_edges_black(ax_y)
    # ax_x.plot(t, x[0, : len(t), 0])
    # ax_y.plot(t, x[0, : len(t), 1])

    # Set up figure
    fig_xy = plt.figure(figsize=(10, 10))
    ax_xy = fig_xy.add_subplot(111)
    set_edges_black(ax_xy)
    lwidth = 5.0
    obstacle1_x, obstacle1_y = get_circle(np.array([-2.5, 0]), 1.0, 100)
    obstacle2_x, obstacle2_y = get_circle(np.array([1.25, -1.25]), 1.0, 100)
    ax_xy.plot(obstacle1_x, obstacle1_y, "k", linewidth=lwidth)
    ax_xy.plot(obstacle2_x, obstacle2_y, "k", linewidth=lwidth, label="Obstacles")
    ax_xy.add_artist(
        Circle(
            (-2.5, 0.0),
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
    ax_xy.plot(
        xref[0, : len(t), 0],
        xref[0, : len(t), 1],
        "--",
        linewidth=lwidth,
        label="Reference",
    )
    ax_xy.plot(
        xnom[0, : len(t), 0],
        xnom[0, : len(t), 1],
        ":",
        linewidth=lwidth,
        label="Nominal Controller",
    )
    ax_xy.plot(
        xnai[0, : len(t), 0],
        xnai[0, : len(t), 1],
        "-.",
        linewidth=lwidth,
        label="Naive CBF Controller",
    )
    ax_xy.plot(
        xrob[0, : len(t), 0], xrob[0, : len(t), 1], linewidth=lwidth, label="Robust CBF Controller"
    )
    ax_xy.plot(
        xrad[0, : len(t), 0],
        xrad[0, : len(t), 1],
        linewidth=lwidth,
        label="Robust-Adaptive CBF Controller",
    )

    ax_inset = inset_axes(
        ax_xy,
        width="100%",
        height="100%",
        bbox_to_anchor=(0.1, 0.025, 0.25, 0.15),
        bbox_transform=ax_xy.transAxes,
        loc=3,
    )
    ax_inset.spines["bottom"].set_color("#000000")
    ax_inset.spines["top"].set_color("#000000")
    ax_inset.spines["right"].set_color("#000000")
    ax_inset.spines["left"].set_color("#000000")

    ax_inset.add_artist(
        Circle(
            (-2.5, 0.0),
            1.0,
            edgecolor="black",
            facecolor=(0, 0, 0),
        )
    )
    ax_inset.add_artist(
        Circle(
            (1.25, -1.25),
            1.0,
            edgecolor="black",
            facecolor=(0, 0, 0),
        )
    )
    ax_inset.plot(
        xref[0, : len(t), 0],
        xref[0, : len(t), 1],
        "--",
        linewidth=lwidth,
        label="Reference",
    )
    ax_inset.plot(
        xnom[0, : len(t), 0],
        xnom[0, : len(t), 1],
        ":",
        linewidth=lwidth,
        label="Nominal Controller",
    )
    ax_inset.plot(
        xnai[0, : len(t), 0],
        xnai[0, : len(t), 1],
        "-.",
        linewidth=lwidth,
        label="Naive CBF Controller",
    )
    ax_inset.plot(
        xrob[0, : len(t), 0], xrob[0, : len(t), 1], linewidth=lwidth, label="Robust CBF Controller"
    )
    ax_inset.plot(
        xrad[0, : len(t), 0],
        xrad[0, : len(t), 1],
        linewidth=lwidth,
        label="Robust-Adaptive CBF Controller",
    )
    ax_inset.set_xlim([-2.9, -2.7])
    ax_inset.set_ylim([-1.0, -0.9])
    ax_inset.set(xticklabels=[], yticklabels=[])
    mark_inset(ax_xy, ax_inset, loc1=3, loc2=1, fc="none", ec="0.2", lw=1.5)

    ax_xy.legend(fancybox=True, fontsize=20)
    ax_xy.set_xlim([-4, 4])
    ax_xy.set_ylim([-3, 4])
    ax_xy.set(xlabel="X", ylabel="Y")
    for item in (
        [ax_xy.title, ax_xy.xaxis.label, ax_xy.yaxis.label]
        + ax_xy.get_xticklabels()
        + ax_xy.get_yticklabels()
    ):
        item.set_fontsize(35)

    return [fig_xy]


def generate_estimation_figures(
    t: NDArray,
    f_nai: List[NDArray],
    f_rob: List[NDArray],
    f_rad: List[NDArray],
) -> List:
    """Generates static figures for plotting the state trajectories.

    Arguments:
        t: time vector
        f_err: time history of the unknown function estimation error

    Returns:
        figs: list to handles of figure objects

    """
    # Parse arguments
    f_err_nai, f_est_nai = f_nai[0], f_nai[1]
    f_err_rob, f_est_rob = f_rob[0], f_rob[1]
    f_err_rad, f_est_rad = f_rad[0], f_rad[1]

    # Set up figure
    lwidth = 5
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    set_edges_black(ax1)
    set_edges_black(ax2)
    # ax1.plot(t, f_err[0, : len(t), 0])
    # ax1.plot(t, f_err[0, : len(t), 1])
    # ax2.plot(t, f_est[0, : len(t), 0], label="Est dx")
    # ax2.plot(t, f_est[0, : len(t), 1], label="Est dy")
    # ax2.plot(t, f_est[0, : len(t), 0] + f_err[0, : len(t), 0], ":", label="Actual dx")
    # ax2.plot(t, f_est[0, : len(t), 1] + f_err[0, : len(t), 1], ":", label="Actual dy")
    # ax2.set_ylim([-2, 2])
    # ax2.legend(fancybox=True)

    # # Error Plots
    # ax1.plot(t, f_err_nai[0, : len(t), 0], label=r"$\Tilde d_1$")
    # ax1.plot(t, f_err_nai[0, : len(t), 1], label=r"$\Tilde d_2$")
    # ax2.plot(t, f_err_rob[0, : len(t), 0], label=r"$\Tilde d_1$")
    # ax2.plot(t, f_err_rob[0, : len(t), 1], label=r"$\Tilde d_2$")

    # Axis 1
    T_fixed = 0.12
    ax1.plot(
        t,
        f_est_nai[0, : len(t), 0] + f_err_nai[0, : len(t), 0],
        label=r"$d_1$ noise-free",
        linewidth=lwidth,
    )
    ax1.plot(
        t,
        f_est_rob[0, : len(t), 0] + f_err_rob[0, : len(t), 0],
        label=r"$d_1$ noisy",
        linewidth=lwidth,
    )
    ax1.plot(
        t, f_est_nai[0, : len(t), 0], ":", label=r"$\hat d_1$ noise-free", linewidth=lwidth - 2
    )
    ax1.plot(t, f_est_rob[0, : len(t), 0], ":", label=r"$\hat d_1$ noisy", linewidth=lwidth - 2)
    ax1.scatter(T_fixed, f_est_nai[0, int(T_fixed * 1000), 0], s=100, label="T")

    # Inset Axis 1
    ax_inset1 = inset_axes(
        ax1,
        width="100%",
        height="100%",
        bbox_to_anchor=(0.05, 0.02, 0.5, 0.5),
        bbox_transform=ax1.transAxes,
        loc=3,
    )
    set_edges_black(ax_inset1)
    ax_inset1.plot(
        t,
        f_est_nai[0, : len(t), 0] + f_err_nai[0, : len(t), 0],
        label=r"$d_1$ nf",
        linewidth=lwidth,
    )
    ax_inset1.plot(
        t, f_est_rob[0, : len(t), 0] + f_err_rob[0, : len(t), 0], label=r"$d_1$ n", linewidth=lwidth
    )
    ax_inset1.plot(
        t, f_est_nai[0, : len(t), 0], ":", label=r"$\hat d_1$ noise-free", linewidth=lwidth - 2
    )
    ax_inset1.plot(
        t, f_est_rob[0, : len(t), 0], ":", label=r"$\hat d_1$ noisy", linewidth=lwidth - 2
    )
    ax_inset1.scatter(T_fixed, f_est_nai[0, int(T_fixed * 1000), 0], s=100)
    ax_inset1.set_xlim([-0.01, 0.2])
    ax_inset1.set_ylim([-3.0, 8.0])
    ax_inset1.set(xticklabels=[], yticklabels=[])
    mark_inset(ax1, ax_inset1, loc1=3, loc2=1, fc="none", ec="0.2", lw=1.5)

    # Axis 2
    ax2.plot(
        t,
        f_est_nai[0, : len(t), 1] + f_err_nai[0, : len(t), 1],
        label=r"$d_2$ noise-free",
        linewidth=lwidth,
    )
    ax2.plot(
        t,
        f_est_rob[0, : len(t), 1] + f_err_rob[0, : len(t), 1],
        label=r"$d_2$ noisy",
        linewidth=lwidth,
    )
    ax2.plot(
        t, f_est_nai[0, : len(t), 1], ":", label=r"$\hat d_2$ noise-free", linewidth=lwidth - 2
    )
    ax2.plot(t, f_est_rob[0, : len(t), 1], ":", label=r"$\hat d_2$ noisy", linewidth=lwidth - 2)
    ax2.scatter(T_fixed, f_est_nai[0, int(T_fixed * 1000), 1], s=100, label="T")

    # Inset Axis 2
    ax_inset2 = inset_axes(
        ax2,
        width="100%",
        height="100%",
        bbox_to_anchor=(0.05, 0.02, 0.5, 0.5),
        bbox_transform=ax2.transAxes,
        loc=3,
    )
    set_edges_black(ax_inset2)
    ax_inset2.plot(
        t,
        f_est_nai[0, : len(t), 1] + f_err_nai[0, : len(t), 1],
        label=r"$d_2$ nf",
        linewidth=lwidth,
    )
    ax_inset2.plot(
        t, f_est_rob[0, : len(t), 1] + f_err_rob[0, : len(t), 1], label=r"$d_2$ n", linewidth=lwidth
    )
    ax_inset2.plot(
        t, f_est_nai[0, : len(t), 1], ":", label=r"$\hat d_2$ noise-free", linewidth=lwidth - 2
    )
    ax_inset2.plot(
        t, f_est_rob[0, : len(t), 1], ":", label=r"$\hat d_2$ noisy", linewidth=lwidth - 2
    )
    ax_inset2.scatter(T_fixed, f_est_nai[0, int(T_fixed * 1000), 1], s=100)
    ax_inset2.set_xlim([-0.01, 0.2])
    ax_inset2.set_ylim([-1.0, 7.0])
    ax_inset2.set(xticklabels=[], yticklabels=[])
    mark_inset(ax2, ax_inset2, loc1=3, loc2=1, fc="none", ec="0.2", lw=1.5)

    ax1.set_xlim([-0.1, 10])
    ax2.set_xlim([-0.1, 10])
    ax1.set_ylim([-85, 20])
    ax2.set_ylim([-45, 20])
    ax1.legend(fancybox=True, loc="lower right", fontsize=20)
    ax2.legend(fancybox=True, loc="lower right", fontsize=20)
    ax1.set(ylabel="d (m\sec)")
    ax2.set(xlabel="Time (sec)", ylabel="d (m/sec)")

    # Font Sizes
    for item in (
        [ax1.title, ax1.xaxis.label, ax1.yaxis.label]
        + ax1.get_xticklabels()
        + ax1.get_yticklabels()
    ):
        item.set_fontsize(35)
    for item in (
        [ax2.title, ax2.xaxis.label, ax2.yaxis.label]
        + ax2.get_xticklabels()
        + ax2.get_yticklabels()
    ):
        item.set_fontsize(35)

    # Figure list
    figs = [fig]

    return figs


def generate_control_figures(
    t: NDArray,
    u_nai: List[NDArray],
    u_rob: List[NDArray],
    u_rad: List[NDArray],
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
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    set_edges_black(ax1)
    set_edges_black(ax2)

    # Control Plots
    ax1.plot(t, u_nai[0, : len(t), 0], label=r"$v_x$ (Naive CBF)", linewidth=lwidth)
    ax1.plot(t, u_rob[0, : len(t), 0], "-.", label=r"$v_x$ (Robust CBF)", linewidth=lwidth)
    ax1.plot(t, u_rad[0, : len(t), 0], ":", label=r"$v_x$ (Robust-Adaptive CBF)", linewidth=lwidth)
    ax2.plot(t, u_nai[0, : len(t), 1], label=r"$v_y$ (Naive CBF)", linewidth=lwidth)
    ax2.plot(t, u_rob[0, : len(t), 1], "-.", label=r"$v_y$ (Robust CBF)", linewidth=lwidth)
    ax2.plot(t, u_rad[0, : len(t), 1], ":", label=r"$v_y$ (Robust-Adaptive CBF)", linewidth=lwidth)

    # ax1.set_xlim([-0.1, 10])
    # ax2.set_xlim([-0.1, 10])
    ax1.set_ylim([-20, 35])
    ax2.set_ylim([-20, 35])
    ax1.legend(fancybox=True, fontsize=20, loc="upper right")
    ax2.legend(fancybox=True, fontsize=20, loc="upper right")
    ax1.set(ylabel=r"$v_x$ (m\sec)")
    ax2.set(xlabel="Time (sec)", ylabel=r"$v_y$ (m/sec)")

    # Font Sizes
    for item in (
        [ax1.title, ax1.xaxis.label, ax1.yaxis.label]
        + ax1.get_xticklabels()
        + ax1.get_yticklabels()
    ):
        item.set_fontsize(35)
    for item in (
        [ax2.title, ax2.xaxis.label, ax2.yaxis.label]
        + ax2.get_xticklabels()
        + ax2.get_yticklabels()
    ):
        item.set_fontsize(35)

    # Figure list
    figs = [fig]

    return figs


def generate_cbf_figures(
    t: NDArray,
    x_nai: List[NDArray],
    x_rob: List[NDArray],
    x_rad: List[NDArray],
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
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(111)
    set_edges_black(ax1)

    # Generate CBF Values
    cx1 = -2.5
    cy1 = 0.0
    cx2 = 1.25
    cy2 = -1.25
    h_nai_1 = (x_nai[0, : len(t), 0] - cx1) ** 2 + (x_nai[0, : len(t), 1] - cy1) ** 2 - 1
    h_nai_2 = (x_nai[0, : len(t), 0] - cx2) ** 2 + (x_nai[0, : len(t), 1] - cy2) ** 2 - 1
    h_rob_1 = (x_rob[0, : len(t), 0] - cx1) ** 2 + (x_rob[0, : len(t), 1] - cy1) ** 2 - 1
    h_rob_2 = (x_rob[0, : len(t), 0] - cx2) ** 2 + (x_rob[0, : len(t), 1] - cy2) ** 2 - 1
    h_rad_1 = (x_rad[0, : len(t), 0] - cx1) ** 2 + (x_rad[0, : len(t), 1] - cy1) ** 2 - 1
    h_rad_2 = (x_rad[0, : len(t), 0] - cx2) ** 2 + (x_rad[0, : len(t), 1] - cy2) ** 2 - 1

    # CBF Plots
    ax1.plot(t, np.zeros((len(t),)), "k", label="Boundary", linewidth=lwidth)
    ax1.plot(t, h_nai_1, label=r"$h_1$ (Naive CBF)", linewidth=lwidth)
    ax1.plot(t, h_nai_2, label=r"$h_2$ (Naive CBF)", linewidth=lwidth)
    ax1.plot(t, h_rob_1, "-.", label=r"$h_1$ (Robust CBF)", linewidth=lwidth)
    ax1.plot(t, h_rob_2, "-.", label=r"$h_2$ (Robust CBF)", linewidth=lwidth)
    ax1.plot(t, h_rad_1, ":", label=r"$h_1$ (Robust-Adaptive CBF)", linewidth=lwidth)
    ax1.plot(t, h_rad_2, ":", label=r"$h_2$ (Robust-Adaptive CBF)", linewidth=lwidth)

    # Inset Axis 1
    ax_inset1 = inset_axes(
        ax1,
        width="100%",
        height="100%",
        bbox_to_anchor=(0.7, 0.05, 0.25, 0.2),
        bbox_transform=ax1.transAxes,
        loc=3,
    )
    set_edges_black(ax_inset1)
    ax_inset1.plot(t, np.zeros((len(t),)), "k", label="Boundary", linewidth=lwidth)
    ax_inset1.plot(t, h_nai_1, label=r"$h_1$ (Naive CBF)", linewidth=lwidth)
    ax_inset1.plot(t, h_nai_2, label=r"$h_2$ (Naive CBF)", linewidth=lwidth)
    ax_inset1.plot(t, h_rob_1, "-.", label=r"$h_1$ (Robust CBF)", linewidth=lwidth)
    ax_inset1.plot(t, h_rob_2, "-.", label=r"$h_2$ (Robust CBF)", linewidth=lwidth)
    ax_inset1.plot(t, h_rad_1, ":", label=r"$h_1$ (Robust-Adaptive CBF)", linewidth=lwidth)
    ax_inset1.plot(t, h_rad_2, ":", label=r"$h_2$ (Robust-Adaptive CBF)", linewidth=lwidth)
    ax_inset1.set_xlim([7.5, 8.5])
    ax_inset1.set_ylim([-0.2, 0.2])
    ax_inset1.set(xticklabels=[], yticklabels=[])
    mark_inset(ax1, ax_inset1, loc1=4, loc2=2, fc="none", ec="0.2", lw=1.5)

    ax1.set_xlim([-0.1, 15])
    ax1.set_ylim([-0.1, 25])
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
