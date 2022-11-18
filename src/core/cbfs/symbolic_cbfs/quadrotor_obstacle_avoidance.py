import builtins
import numpy as np
import symengine as se
from typing import Callable
from nptyping import NDArray
from importlib import import_module
from core.cbfs.cbf_wrappers import symbolic_cbf_wrapper_singleagent
from core.mathematics.symbolic_functions import ramp

vehicle = builtins.PROBLEM_CONFIG["vehicle"]
control_level = builtins.PROBLEM_CONFIG["control_level"]
mod = "models." + vehicle + "." + control_level + ".system"

# Programmatic import
try:
    module = import_module(mod)
    globals().update({"f": getattr(module, "f")})
    globals().update({"g": getattr(module, "g")})
    globals().update({"ss": getattr(module, "xs")})
except ModuleNotFoundError as e:
    print("No module named '{}' -- exiting.".format(mod))
    raise e

# Defining Physical Params
R = 1.0
cx1 = -2.0
cy1 = 0.0
cx2 = 1.25
cy2 = -1.25

# Define Physical Distances
dx1 = ss[0] - cx1
dy1 = ss[1] - cy1
dx2 = ss[0] - cx2
dy2 = ss[1] - cy2

# Symbolic drift and control vectors
f_sym = f(np.zeros((len(ss),)), True)
g_sym = g(np.zeros((len(ss),)), True)

# Define more quantities
phidot = f_sym[6]
thedot = f_sym[7]


class QuadrotorObstacleAvoidanceCbf:
    """Object for implementing 3rd order Quadrotor (cascaded) CBF."""

    def __init__(self, syms: se.symbols, sym_cbf: se.symbols, f_sym: se.symbols, g_sym: se.symbols):
        self.sym_state = syms
        self.f_sym = f_sym
        self.g_sym = g_sym

        # Build symbolic CBF and partial derivatives
        self.h_sym = sym_cbf
        self.dhdx_sym = (se.DenseMatrix([self.h_sym]).jacobian(se.DenseMatrix(self.sym_state))).T
        self.d2hdx2_sym = self.dhdx_sym.jacobian(se.DenseMatrix(self.sym_state))

        # Build executable CBF and partial derivatives
        self.h = symbolic_cbf_wrapper_singleagent(self.h_sym, self.sym_state)
        self.dhdx = symbolic_cbf_wrapper_singleagent(self.dhdx_sym, self.sym_state)
        self.d2hdx2 = symbolic_cbf_wrapper_singleagent(self.d2hdx2_sym, self.sym_state)

        # Build executable dynamics functions
        self.f = f
        self.g = g

        # Define Exponential CBF Weights (roots of
        # s^n + lambda_{n-1}*s^{n-1} + ... + lambda_1*s + lambda_0 = 0
        # must have real parts negative)
        self.lambda_0 = 1.0
        self.lambda_1 = 3.0
        self.lambda_2 = 3.0

    def hdot_sym(self, force: float) -> se.symbols:
        """Computes the symbolic time derivative of the h function.

        Arguments:
            x: state vector at current time
            force: computed force input

        Returns:
            hdot: time derivative of h function

        """
        return self.dhdx_sym.T @ (self.f_sym + force * self.g_sym[:, 0])

    def h2dot_sym(self, force: float) -> se.symbols:
        """Computes the symbolic 2nd time derivative of the h function.

        Arguments:
            force: force control input

        Returns
            h2dot: symbolic

        """
        dhdotdx = self.hdot_sym(force).jacobian(se.DenseMatrix(self.sym_state))

        return dhdotdx @ self.f_sym

    def hdot(self, x: NDArray, force: float) -> float:
        """Computes the time derivative of the h function.

        Arguments:
            x: state vector at current time
            force: computed force input

        Returns:
            hdot: time derivative of h function

        """
        to_eval = symbolic_cbf_wrapper_singleagent(self.hdot_sym(force), self.sym_state)

        return np.squeeze(np.array(to_eval(x)).astype(np.float64))

    def h2dot(self, x: NDArray, force: float) -> float:
        """Computes the 2nd time-derivative of the function h.

        Arguments:
            x: state vector

        Returns:
            h2dot: 2nd time derivative

        """
        to_eval = symbolic_cbf_wrapper_singleagent(self.h2dot_sym(force), self.sym_state)

        return np.squeeze(np.array(to_eval(x)).astype(np.float64))

    def C_sym(self, force: float) -> se.symbols:
        """Symbolic 3rd order CBF.

        Arguments:
            force

        Returns:
            cbf_sym

        """
        return (
            self.h2dot_sym(force)[0]
            + self.lambda_1 * self.hdot_sym(force)[0]
            + self.lambda_0 * self.h_sym
        )

    def dCdx_sym(self, force: float) -> se.symbols:
        """Partial derivative of c function wrt state x.

        Arguments:
            force: force input

        Returns:
            symbolic dCdx

        """
        return se.DenseMatrix([self.C_sym(force)]).jacobian(se.DenseMatrix(self.sym_state))

    def C(self, x: NDArray, force: float) -> float:
        """Computes the value of the cbf at the state and force input.

        Arguments:
            x: state
            force: force

        Returns:
            cbf value

        """
        to_eval = symbolic_cbf_wrapper_singleagent(self.C_sym(force), self.sym_state)

        return np.squeeze(np.array(to_eval(x)).astype(np.float64))

    def dCdx(self, x: NDArray, force: float) -> NDArray:
        """Computes the value of the cbf at the state and force input.

        Arguments:
            x: state vector
            force: force

        Returns:
            cbf value

        """
        to_eval = symbolic_cbf_wrapper_singleagent(self.dCdx_sym(force), self.sym_state)

        return np.squeeze(np.array(to_eval(x)).astype(np.float64))


# Instantiate CBF objects
cbf_oa1 = QuadrotorObstacleAvoidanceCbf(ss, dx1**2 + dy1**2 - R**2, f_sym, g_sym)
cbf_oa2 = QuadrotorObstacleAvoidanceCbf(ss, dx2**2 + dy2**2 - R**2, f_sym, g_sym)


# Test
x = np.zeros((12,))
force = 9.81
print(cbf_oa1.C(x, force))
print(cbf_oa2.C(x, force))
print(cbf_oa1.dCdx(x, force))
print(cbf_oa2.dCdx(x, force))

# Define CBF Functions
def h_oa1(x: NDArray, force: float) -> float:
    """Function to evaluate the CBF1 at state x.

    Arguments:
        x: state vector
        force: force input

    Returns:
        cbf val

    """
    return cbf_oa1.C(x, force)


def h_oa2(x: NDArray, force: float) -> float:
    """Function to evaluate the CBF2 at state x.

    Arguments:
        x: state vector
        force: force input

    Returns:
        cbf val

    """
    return cbf_oa2.C(x, force)


# Define CBF partial derivative functions
def dhdx_oa1(x: NDArray, force: float) -> float:
    """Function to evaluate the CBF1 at state x.

    Arguments:
        x: state vector
        force: force input

    Returns:
        cbf val

    """
    return cbf_oa1.dCdx(x, force)


def dhdx_oa2(x: NDArray, force: float) -> float:
    """Function to evaluate the CBF2 at state x.

    Arguments:
        x: state vector
        force: force input

    Returns:
        cbf val

    """
    return cbf_oa2.dCdx(x, force)


def d2hdx2_oa1(x: NDArray, force: float) -> float:
    """Dummy."""
    return 0


def d2hdx2_oa2(x: NDArray, force: float) -> float:
    """Dummy."""
    return 0
