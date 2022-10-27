"""fxt_adaptation_cbf_qp_controller.py

This file provides the interface to the FxtAdaptationCbfQpController object,
which computes a control input using a synthesis of the well-studied control
barrier function (CBF) based quadratic program (QP) controller and fixed-time
Koopman matrix estimation.

"""
import builtins
from typing import Callable, List, Tuple
from importlib import import_module
import numpy as np
import numdifftools as nd
from nptyping import NDArray
from scipy.linalg import block_diag, null_space, logm

# from core.cbfs.cbf import Cbf
from core.controllers.cbf_qp_controller import CbfQpController
from core.controllers.controller import Controller

# from core.solve_cvxopt import solve_qp_cvxopt

vehicle = builtins.PROBLEM_CONFIG["vehicle"]
control_level = builtins.PROBLEM_CONFIG["control_level"]
system_model = builtins.PROBLEM_CONFIG["system_model"]
mod = "models." + vehicle + "." + control_level + ".system"

# Programmatic import
module = import_module(mod)
globals().update({"f": getattr(module, "f")})
globals().update({"g": getattr(module, "g")})
globals().update({"sigma": getattr(module, "sigma_{}".format(system_model))})


def basis_functions(z: NDArray, min_len: int) -> NDArray:
    """Computes the values of the basis functions evaluated at the current
    state and control values. Returns the (b x 1) vector of basis function
    values. May offload this to another module.

    Arguments
        z: input vector (may be states and controls or outputs)
        min_len: minimum length of input vector

    Returns
        basis_funcs: vector of values of basis functions

    """
    # Append zeros to input z if necessary
    if len(z) < min_len:
        z = np.concatenate([z, np.zeros((min_len - len(z),))])

    # Monomial basis functions
    psi_0nn = z  # 1st Order
    psi_1nn = z**2  # 2nd order
    psi_2nn = z**3  # 3rd Order
    psi_3nn = z**4  # 4th Order
    psi_4nn = z**5  # 5th Order

    # Radial Basis Functions (RBFs)
    k = 10.0  # Multiplier for RBF
    Q = 1 / k * np.eye(len(z))  # Exponential gain for RBF

    psi_5n1 = k * np.exp(-1 / 2 * (z @ Q @ z))  # Radial Basis Functions
    psi_6nn = -k * Q @ z * np.exp(-1 / 2 * (z @ Q @ z))  # Gradient of RBF wrt z

    basis_funcs = np.hstack(
        [
            psi_0nn,
            psi_1nn,
            psi_2nn,
            psi_3nn,
            psi_4nn,
            psi_5n1,
            psi_6nn,
        ]
    )

    return basis_funcs


class FxtAdaptationCbfQpController(CbfQpController):
    """
    Adaptation-based CBF-QP controller for systems subject to unknown, nonlinear,
    additive disturbance in the system dynamics.

    Public Methods:
        update_parameter_estimates
        compute_theta_dot

    Class Properties:
        TBD
    """

    # NOTE: The initial eta parameters can be computed using initial parameter
    # values of zero and by assuming that the function being estimated is
    # bounded by some value L.

    _updated = False
    _eta = 100.0  #! Lazy coding -- needs to be fixed
    _etadot = 0.0  #! Lazy coding -- needs to be fixed
    _nu = -100.0  #! Lazy coding -- needs to be fixed

    def __init__(
        self,
        u_max: List,
        nAgents: int,
        objective_function: Callable,
        nominal_controller: Controller,
        cbfs_individual: List,
        cbfs_pairwise: List,
        ignore: List = None,
    ):
        super().__init__(
            u_max,
            nAgents,
            objective_function,
            nominal_controller,
            cbfs_individual,
            cbfs_pairwise,
            ignore,
        )
        n_cbfs = len(self.cbf_vals)
        self.n_agents = 1  #! Lazy coding, need to fix
        self.n_states = 12  #! Lazy coding, need to fix

        # Unknown parameter properties
        example_basis_fcns = basis_functions(
            np.zeros((self.n_states + len(u_max),)), self.n_states + len(u_max)
        )
        self.n_params = len(example_basis_fcns) ** 2
        self.theta = np.zeros((self.n_params,))
        self.theta_dot = np.zeros((self.n_params,))
        self.eta0 = 100.0 * np.ones((self.n_params,))  #! Lazy coding -- needs to be fixed
        self.M = np.zeros((len(example_basis_fcns), self.n_params))

        # Gains
        self.law_gains = {"a": 1.0, "b": 1.0, "w": 5.0, "G": np.eye(self.n_params)}

        # Miscellaneous parameters
        self.safety = True
        self.max_class_k = 1e6  # Maximum allowable gain for linear class K function
        self.nominal_class_k = 1.0  # Nominal value for linear class K function
        self.discretization_error = 1.0  #! This needs to be corrected
        self.regressor = np.zeros((self.n_states, self.n_params))

        # CBF Parameters
        self.h = 100 * np.ones((n_cbfs,))
        self.h0 = 100 * np.ones((n_cbfs,))
        self.dhdx = np.zeros((n_cbfs, 2 * self.n_states))  #! Lazy coding -- need to fix!!!
        self.Lfh = np.zeros((n_cbfs,))
        self.Lgh = np.zeros((n_cbfs, 4))  #! Lazy coding -- need to fix!!!

    def formulate_qp(
        self, t: float, ze: NDArray, zr: NDArray, u_nom: NDArray, ego: int, cascade: bool = False
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
        """Formulates the quadratic program objective function matrix Q and
        vector p, the inequality constraint matrix A and vector b, and equality
        constraint matrix G and vector h to be fed into the solve_qp function
        to generate the control input, i.e.

        min J = 1/2 * x.T @ Q @ x + p @ x
        s.t.
        Ax <= b
        Gx = h

        Arguments:
            t: time in sec
            ze: array containing ego state
            zr: array containing states of remaining (non-ego) agents
            u_nom: nominal control input for agent in question
            ego: identifier of ego agent

        Returns:
            Q: positive-definite matrix for quadratic term of objective function
            p: vector for linear term of objective function
            A: matrix multiplying decision variables for affine inequality constraint
            b: vector for affine inequality constraint
            G: matrix multiplying decision variables for affine equality constraint
            h: vector for affine equality constraint

        """
        # Update state variable
        self.z_ego = ze

        # Update Parameter Estimates
        if t > 0:
            theta, theta_dot = self.update_parameter_estimates()

        # Update last measured state
        self.z_ego_last = self.z_ego

        # Compute Q matrix and p vector for QP objective function
        Q, p = self.compute_objective_qp(u_nom)

        # Compute input constraints of form Au @ u <= bu
        Au, bu = self.compute_input_constraints()

        # Compute individual CBF constraints
        Ai, bi = self.compute_individual_cbf_constraints(ze, ego)

        # Compute pairwise/interagent CBF constraints
        Ap, bp = self.compute_pairwise_cbf_constraints(ze, zr, ego)

        A = np.vstack([Au, Ai, Ap])
        b = np.hstack([bu, bi, bp])

        return Q, p, A, b, None, None

    def compute_objective_qp(self, u_nom: NDArray) -> (NDArray, NDArray):
        """Computes the matrix Q and vector p for the objective function of the
        form

        J = 1/2 * x.T @ Q @ x + p @ x

        Arguments:
            u_nom: nominal control input for agent in question

        Returns:
            Q: quadratic term positive definite matrix for objective function
            p: linear term vector for objective function

        """
        if self.n_dec_vars > 0:
            Q, p = self.objective(np.append(u_nom.flatten(), self.nominal_class_k))
        else:
            Q, p = self.objective(u_nom.flatten())

        return Q, p

    def compute_input_constraints(self):
        """
        Computes matrix Au and vector bu encoding control input constraints of
        the form

        Au @ u <= bu

        Arguments:
            None

        Returns:
            Au: input constraint matrix
            bu: input constraint vector

        """
        if self.n_dec_vars > 0:
            Au = block_diag(*(self.n_agents + self.n_dec_vars) * [self.au])[
                : -2 * (self.n_controls - 1), : -(self.n_controls - 1)
            ]
            bu = np.append(
                np.array(self.n_agents * [self.bu]).flatten(),
                self.n_dec_vars * [self.max_class_k, 0],
            )

        else:
            Au = block_diag(*(self.n_agents) * [self.au])
            bu = np.array(self.n_agents * [self.bu]).flatten()

        return Au, bu

    def compute_individual_cbf_constraints(self, ze: NDArray, ego: int) -> (NDArray, NDArray):
        """Computes matrix Ai and vector bi for individual CBF constraints
        of the form

        Ai @ u <= bi

        where an individual cbf constraint is defined as one for which safety
        is dependent only on the ego agent, not the behavior of other agents
        present in the system.

        Arguments:
            ze: ego state
            ego: identifier of ego agent

        Returns:
            Ai: individual cbf constraint matrix (nCBFs x nControls)
            bi: individual cbf constraint vector (nCBFs x 1)

        """
        # Parameters
        ns = len(ze)  # Number of states
        Ai = np.zeros(
            (len(self.cbfs_individual), self.n_controls * self.n_agents + self.n_dec_vars)
        )
        bi = np.zeros((len(self.cbfs_individual),))

        # Iterate over individual CBF constraints
        for cc, cbf in enumerate(self.cbfs_individual):
            self.h0[cc] = cbf.h0(ze)
            self.h[cc] = cbf.h(ze)
            self.dhdx[cc][:ns] = cbf.dhdx(ze)
            dhdx = self.dhdx[cc][:ns]

            # Stochastic Term -- 0 for deterministic systems
            if np.trace(sigma(ze).T @ sigma(ze)) > 0 and self._stochastic:
                d2hdx2 = cbf.d2hdx2(ze)
                stoch = 0.5 * np.trace(sigma(ze).T @ d2hdx2 @ sigma(ze))
            else:
                stoch = 0.0

            # Time-Varying Parameter Term
            tv_term = self.compute_time_varying_cbf_term(dhdx)

            # Get CBF Lie Derivatives
            self.Lfh[cc] = dhdx @ f(ze) + stoch - self.discretization_error + tv_term
            self.Lgh[cc, self.n_controls * ego : (ego + 1) * self.n_controls] = dhdx @ g(
                ze
            )  # Only assign ego control

            # Generate Ai and bi from CBF and associated derivatives
            Ai[cc, :], bi[cc] = self.generate_cbf_condition(
                cbf, self.h[cc], self.Lfh[cc], self.Lgh[cc], cc, adaptive=True
            )

        # Check whether any of the safety conditions are violated
        if (np.sum(self.h0[: len(self.cbfs_individual)] < 0) > 0) or (
            np.sum(self.h[: len(self.cbfs_individual)] < 0) > 0
        ):
            self.safety = False
            print(f"Safety Violation!! h0 --> {np.min(self.h0[:len(self.cbfs_individual)])}")
            print(f"Safety Violation!! h  --> {np.min(self.h[:len(self.cbfs_individual)])}")

        return Ai, bi

    def compute_pairwise_cbf_constraints(
        self, ze: NDArray, zr: NDArray, ego: int
    ) -> (NDArray, NDArray):
        """Computes matrix Ap and vector bp for paiwise CBF constraints
        of the form

        Ap @ u <= bp

        where a paairwise cbf constraint is defined as one for which safety
        is dependent on both the ego agent and the behavior of some other
        agents present in the system (e.g. collision avoidance).

        Arguments:
            ze: ego state
            zr: array of states of remaining agents
            ego: identifier of ego agent

        Returns:
            Ap: pairwise cbf constraint matrix (nCBFs x nControls)
            bp: pairwise cbf constraint vector (nCBFs x 1)

        """
        ns = len(ze)  # Number of states
        lci = len(self.cbfs_individual)
        Ap = np.zeros(
            (len(self.cbfs_pairwise) * len(zr), self.n_controls * self.n_agents + self.n_dec_vars)
        )
        bp = np.zeros((len(self.cbfs_pairwise) * len(zr),))

        # Iterate over pairwise CBF constraints
        for cc, cbf in enumerate(self.cbfs_pairwise):

            # Iterate over all other vehicles
            for ii, zo in enumerate(zr):
                other = ii + (ii >= ego)
                idx = lci + cc * zr.shape[0] + ii

                self.h0[idx] = cbf.h0(ze, zo)
                self.h[idx] = cbf.h(ze, zo)
                self.dhdx[idx] = cbf.dhdx(ze, zo)
                dhdx = self.dhdx[idx]

                # Stochastic Term -- 0 for deterministic systems
                if np.trace(sigma(ze).T @ sigma(ze)) > 0 and self._stochastic:
                    d2hdx2 = cbf.d2hdx2(ze, zo)
                    stoch = 0.5 * (
                        np.trace(sigma(ze).T @ d2hdx2[:ns, :ns] @ sigma(ze))
                        + np.trace(sigma(zo).T @ d2hdx2[ns:, ns:] @ sigma(zo))
                    )
                else:
                    stoch = 0.0

                # Time-Varying Parameter Term
                tv_term = self.compute_time_varying_cbf_term(dhdx)

                # Get CBF Lie Derivatives
                self.Lfh[idx] = (
                    dhdx[:ns] @ f(ze)
                    + dhdx[ns:] @ f(zo)
                    + stoch
                    - self.discretization_error
                    + tv_term
                )
                self.Lgh[idx, self.n_controls * ego : (ego + 1) * self.n_controls] = dhdx[:ns] @ g(
                    ze
                )
                self.Lgh[idx, self.n_controls * other : (other + 1) * self.n_controls] = dhdx[
                    ns:
                ] @ g(zo)

                # Generate Ap and bp from CBF and associated derivatives
                p_idx = cc * len(self.cbfs_pairwise) + ii
                Ap[p_idx, :], bp[p_idx] = self.generate_cbf_condition(
                    cbf, self.h[idx], self.Lfh[idx], self.Lgh[idx], idx, adaptive=True
                )

                # Check whether any of the safety conditions are violated
        if (np.sum(self.h0[len(self.cbfs_individual) :] < 0) > 0) or (
            np.sum(self.h[len(self.cbfs_individual) :] < 0) > 0
        ):
            self.safety = False
            print(f"Safety Violation!! h0 --> {np.min(self.h0[len(self.cbfs_individual):])}")
            print(f"Safety Violation!! h  --> {np.min(self.h[len(self.cbfs_individual):])}")

        return Ap, bp

    def update_parameter_estimates(self) -> Tuple[NDArray, NDArray]:
        """Updates parameters comprising the approximated Koopman Operator
        according to the following parameter update law:

        thetadot = Gamma @ M.T @ v * (a * ||v||^(2 / u) + b / ||v||^(2 / u))

        where M and v are related according to Mz = v, with z the parameter
        estimation error.

        Arguments:
            TBD

        Returns:
            theta: updated parameter estimates in system dynamics
            thetadot: time-derivative of parameter estimates according to
                adaptation law

        """
        # Compute time-derivatives of theta parameters
        self.theta_dot = self.compute_theta_dot()

        # Update theta parameters according to first-order forward-Euler
        self.theta = self.theta + self.theta_dot * self._dt

        return self.theta, self.theta_dot

    #! TO DO: Implement measurements as outputs
    def compute_theta_dot(self) -> NDArray:
        """Computes the time-derivative of the Koopman parameters according to
        the following parameter update law:

        thetadot = Gamma @ M.T @ v * (a * ||v||^(2 / u) + b / ||v||^(2 / u))

        where M and v are related according to Mz = v, with z the parameter
        estimation error.

        Arguments:
            TBD

        Returns:
            theta_dot: time-derivative of parameter estimates in system dynamics

        """
        # Generate Px and Py from input/output data
        px = self.compute_basis_functions(np.concatenate([self.z_ego, self.u]))
        py = self.compute_basis_functions(self.outputs)

        # Compute matrix M and vector v for adaptation law
        self.M = block_diag(*(len(px)) * [px])
        v = py - self.M @ self.theta

        a = self.law_gains["a"]
        b = self.law_gains["b"]
        w = self.law_gains["w"]
        G = self.law_gains["G"]
        norm_v = np.linalg.norm(v)

        theta_dot = G @ self.M.T @ v * (a * norm_v ** (2 / w) + b / norm_v ** (2 / w))

        return theta_dot

    def compute_time_varying_cbf_term(self, dhdx: NDArray) -> float:
        """Computes the contribution of the time-varying parameters in the
        system dynamics to the time-derivative of the CBF under consideration.

        Arguments:
            dhdx: partial derivative of CBF h with respect to state x

        Returns:
            tv_term: time-varying term in CBF derivative based on theta dot

        """
        G = self.law_gains["G"]
        tv_term = np.trace(G) * self.eta * self.etadot + self.compute_nu(dhdx)

        return tv_term

    #! TO DO: Finish handling regressor matrix and state
    def compute_nu(self, dhdx: NDArray) -> float:
        """Computes nu, the effect of the worst-case admissible parameters on
        the evolution of the CBF trajectories.

        Arguments:
            None

        Returns:
            nu: worst-case effect of unknown parameters on the CBF trajectories

        """
        Ldh = dhdx @ self.regressor
        element_wise_min = np.minimum(Ldh * (self.theta + self.eta), Ldh * (self.theta - self.eta))
        nu = np.sum(element_wise_min)

        return nu

    def compute_eta_and_etadot(self) -> Tuple[float, float]:
        """Computes both eta (the maximum parameter estimation error) and
        etadot (its time-derivative) according to the adaptation law (and gains).

        Arguments:
            None

        Returns:
            eta: maximum parameter estimation errror
            etadot: time-derivative of maximum parameter estimation error

        """
        # Import gains
        a = self.law_gains["a"]
        b = self.law_gains["b"]
        G = self.law_gains["G"]
        w = self.law_gains["w"]

        # Compute derived parameters
        null_m = null_space(self.M)
        rank_m = null_m.shape[0] - null_m.shape[1]
        _, sigmas, _ = np.linalg.svd(self.M)
        sigma_r = sigmas[rank_m - 1] if rank_m > 0 else 0
        kv = sigma_r * np.sqrt(2 * np.max(G))
        c1 = a * kv ** (2 + 2 / w)
        c2 = b * kv ** (2 - 2 / w)
        Xi = np.arctan2(
            np.sqrt(c2) * (1 / 2 * self.eta0 @ np.linalg.inv(G) @ self.eta0) ** (1 / w),
            np.sqrt(c1),
        )
        A = np.max([Xi - np.sqrt(c1 * c2) / w * self.t, 0])

        # Compute eta
        eta = np.sqrt(2 * np.max(G) * (np.sqrt(c1 / c2) * np.tan(A)) ** (w))

        # Compute etadot
        etadot = (
            -c1
            * np.sqrt(np.max(G) / 2)
            * (np.sqrt(c1 / c2) * np.tan(A)) ** (w / 2 - 1)
            / np.cos(A) ** 2
        )

        return eta, etadot

    def compute_basis_functions(self, z: NDArray) -> NDArray:
        """Computes the values of the basis functions evaluated at the current
        state and control values. Returns the (b x 1) vector of basis function
        values.

        Arguments
            z: input vector (may be states and controls or outputs)

        Returns
            basis_functions: vector of values of basis functions

        """
        return basis_functions(z, len(self.z_ego) + len(self.u))

    def get_koopman_matrix(self) -> NDArray:
        """Retrieves the (approximated) Koopman matrix from the parameter
        estimates theta.

        Arguments
            None

        Returns
            U: (approximate) Koopman operator

        """
        dim_U = int(np.sqrt(len(self.theta)))
        U = self.theta.reshape((dim_U, dim_U)).T

        return U

    def compute_koopman_generator(self) -> NDArray:
        """Computes the approximate infinitesimal generator L of the
        Koopman Operator U.

        Arguments
            TBD

        Returns
            L: (approximate) infinitesimal generator of Koopman operator

        """
        U = self.get_koopman_matrix()
        L = 1 / self._dt * logm(U)

        return L

    #! TO DO: implement n_states as class variable
    def compute_unknown_function(self) -> NDArray:
        """Computes the approximate infinitesimal generator L of the
        Koopman Operator U.

        Arguments
            TBD

        Returns
            unknown_f_estimate: estimated unknown nonlinear function

        """
        z = np.concatenate([self.z_ego, self.u])
        unknown_f_estimate = (
            self.compute_basis_functions(z) @ self.compute_koopman_generator()[:, : self.n_states]
        )

        return unknown_f_estimate

    @property
    def eta(self):
        """Property for either computing (if necessary to update) or returning
        the value stored for eta, the worst-case parameter estimation error."""
        if not self._updated:
            eta, etadot = self.compute_eta_and_etadot()

            self._eta = eta
            self._etadot = etadot
            self._updated = True

        return self._eta

    @property
    def etadot(self):
        """Property for either computing (if necessary to update) or returning
        the value stored for etadot, the time-derivative of the worst-case
        parameter estimation error."""
        if not self._updated:
            eta, etadot = self.compute_eta_and_etadot()

            self._eta = eta
            self._etadot = etadot
            self._updated = True

        return self._etadot

    @property
    def nu(self):
        """Property for either computing (if necessary to update) or returning
        the value stored for nu, the effect of the worst-case admissible
        parameters on the evolution of the CBF trajectories."""
        if not self._updated:
            eta, etadot = self.compute_eta_and_etadot()

            self._eta = eta
            self._etadot = etadot
            self._updated = True

        return self._nu

    @property
    def outputs(self):
        """Property for computing 'output' for estimator. In this case, the
        estimator seeks to estimate the unknown residual dynamics in the
        system."""

        xdot = (self.z_ego - self.z_ego_last) / self._dt
        outputs = xdot - f(self.z_ego) - g(self.z_ego) @ self.u

        return outputs
