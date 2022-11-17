"""objective_functions.py

Defines methods for objective functions to be used in
quadratic program based control laws.

Contains:
    objective_minimum_deviation
    objective_minimum_norm

"""

from typing import Tuple
from nptyping import NDArray
import numpy as np
from .physical_params import U_MAX


q = [1, 1, 1, 1]
# q0 = np.array([1 / U_MAX[ii] ** 2 for ii in range(len(U_MAX))])
q0 = np.array([q[ii] for ii in range(len(U_MAX))])
q1 = 100


def objective_minimum_deviation(u_nom: NDArray) -> Tuple[NDArray, NDArray]:
    """Defines a minimum-deviation objective function that is quadratic in the
    decision variables, of the form

    J = 1 / 2 * (x - x0).T @ Q @ (x - x0) + p @ (x - x0)

    where x represents the decision variable and x0 is some desired solution.

    Arguments:
        u_nom: nominal control input

    Returns:
        Q: matrix for quadratic term
        p: vector for linear term

    """
    if len(u_nom) == len(U_MAX):
        Q = 1 / 2 * np.diag(q0)
    else:
        qa = (len(u_nom) - len(U_MAX)) * [q1]
        Q = 1 / 2 * np.diag(list(q0) + qa)

    p = -Q @ u_nom

    return Q, p


def objective_minimum_norm(n_vars: int) -> Tuple[NDArray, NDArray]:
    """Defines a minimum-norm objective function that is quadratic in the
    decision variables, of the form

    J = 1 / 2 * x.T @ Q @ x + p @ x

    where x represents the decision variable.

    Arguments:
        n_vars: number of decision variables

    Returns:
        Q: matrix for quadratic term
        p: vector for linear term

    """
    if n_vars % len(U_MAX) == 0:
        Q = 1 / 2 * np.diag(q0)
    else:
        Q = 1 / 2 * np.diag(list(q0) + [q1])

    p = np.zeros((Q.shape[0],))

    return Q, p
