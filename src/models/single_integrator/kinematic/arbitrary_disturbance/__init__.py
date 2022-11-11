"""__init__.py

Loads parameters and functions for the single_integrator/kinematic/wind_field situation.

"""
import platform
import builtins
from importlib import import_module
from core.agent import Agent
from core.cbfs import cbfs_individual, cbfs_pairwise, cbf0
from core.controllers.cbf_qp_controller import CbfQpController
from core.controllers.fxt_adaptation_cbf_qp_controller import FxtAdaptationCbfQpController
from ..system import f, g, nControls
from .timing_params import *
from .physical_params import U_MAX
from .objective_functions import objective_minimum_deviation
from .nominal_controllers import LemniscateTrackingController
from .initial_conditions import z0, u0, N_AGENTS, N_STATES, N_CONTROLS

if builtins.PROBLEM_CONFIG["system_model"] == "stochastic":
    from ..system import (
        sigma_stochastic as sigma,
        stochastic_dynamics as system_dynamics,
        stochastic_step as step_dynamics,
    )
else:
    from ..system import (
        sigma_deterministic as sigma,
        deterministic_dynamics as system_dynamics,
        deterministic_step as step_dynamics,
    )

# Configure parameters
time = [dt, tf]

if platform.machine() == "aarch64":
    save_path = "/home/6lackmitchell/Documents/datastore/warehouse/test.pkl"
else:
    save_path = "/Users/mblack/Documents/git/nonlinear-fxt-adaptation-control/data/single_integrator/kinematic/wind_field/test.pkl"


# Define controllers
def fxt_adaptation_cbf_qp_controller(idx: int) -> FxtAdaptationCbfQpController:
    """Returns instance of a FxtAdaptationCbfQpController object for agent
    corresponding to identifier idx.

    Arguments
        idx: agent identifier

    Returns
        FxtAdaptationCbfQpController

    """
    return FxtAdaptationCbfQpController(
        U_MAX,
        N_AGENTS,
        N_STATES,
        objective_minimum_deviation,
        LemniscateTrackingController(idx),
        cbfs_individual,
        cbfs_pairwise,
    )


# Define CBF Controlled Agents
cbf_controlled_agents = [
    Agent(
        i, z0[i, :], u0, cbf0, time, step_dynamics, fxt_adaptation_cbf_qp_controller(i), save_path
    )
    for i in range(1)
]
# human_agents = [
#     Agent(i, z0[i, :], u0, cbf0, time, step_dynamics, ZeroController(i), save_path)
#     for i in range(3, 9)
# ]
human_agents = []


centralized_agents = None
decentralized_agents = cbf_controlled_agents + human_agents
