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
from nptyping import NDArray
from scipy.linalg import block_diag, null_space

from core.solve_cvxopt import solve_qp_cvxopt
from core.controllers.controller import Controller
from core.controllers.cbf_qp_controller import CbfQpController
from core.mathematics.basis_functions import basis_functions
from core.estimators.koopman_estimators import (
    KoopmanGeneratorEstimator,
)

vehicle = builtins.PROBLEM_CONFIG["vehicle"]
control_level = builtins.PROBLEM_CONFIG["control_level"]
system_model = builtins.PROBLEM_CONFIG["system_model"]
mod = "models." + vehicle + "." + control_level + ".system"

# Programmatic import
module = import_module(mod)
globals().update({"f": getattr(module, "f")})
globals().update({"g": getattr(module, "g")})
globals().update({"sigma": getattr(module, "sigma_{}".format(system_model))})
globals().update({"f_residual": getattr(module, "f_residual")})
globals().update({"g_residual": getattr(module, "g_residual")})


class CascadedCbfQpController(CbfQpController):
    """

    Public Methods:
        update_parameter_estimates
        compute_theta_dot

    Class Properties:
        TBD
    """

    def __init__(
        self,
        u_max: List,
        u_min: List,
        nAgents: int,
        nStates: int,
        objective_function: Callable,
        nominal_controller: Controller,
        cbfs_individual: List,
        cbfs_pairwise: List,
        ignore: List = None,
    ):
        super().__init__(
            u_max,
            u_min,
            nAgents,
            objective_function,
            nominal_controller,
            cbfs_individual,
            cbfs_pairwise,
            ignore,
        )
        n_cbfs = len(self.cbf_vals)
        self.n_agents = nAgents
        self.n_states = nStates

        # Unknown parameter properties
        n_params = len(basis_functions(np.zeros((self.n_states,)))) ** 2
        self.eta0 = 100.0 * np.ones((n_params,))  #! Lazy coding -- needs to be fixed
        self.vector_field_estimation = np.zeros((self.n_states,))
        self.function_estimation_error = np.zeros((self.n_states,))
        self.function_estimation = np.zeros((self.n_states,))

        # Instantiate estimator
        # self.estimator = KoopmanMatrixEstimator(nStates, self._dt, f, g)
        self.estimator = KoopmanGeneratorEstimator(nStates, self._dt, f, g, use_rnn=False)
        # self.estimator = DataDrivenKoopmanMatrixEstimator(nStates, self._dt, f, g, use_rnn=False)

        # Miscellaneous parameters
        self.safety = True
        self.discretization_error = 1.0  #! This needs to be corrected
        self.regressor = np.zeros((self.n_states, n_params))
        self.force = 0

        # CBF Parameters
        self.h = 100 * np.ones((n_cbfs,))
        self.h0 = 100 * np.ones((n_cbfs,))
        self.dhdx = np.zeros((n_cbfs, 2 * self.n_states))
        self.Lfh = np.zeros((n_cbfs,))
        # self.Lgh = np.zeros((n_cbfs, 2))  #! SINGLE INTEGRATOR
        self.Lgh = np.zeros((n_cbfs, 4))  #! QUADROTOR

    def _compute_control(self, t: float, z: NDArray) -> Tuple[NDArray, NDArray, int, str, float]:
        """Computes the vehicle's control input based on a cascaded approach: first, the CBF constraints attempt to
        filter out unsafe inputs on the first level. If no safe control exists, then all control inputs are eligible
        for safety filtering.

        INPUTS
        ------
        t: time (in sec)
        z: full state vector for all vehicles
        extras: anything else

        OUTPUTS
        ------
        u_act: actual control input used in the system
        u_nom: nominal input used if safety not considered
        code: error/success code
        status: more info on error/success

        """
        code = 0
        status = "Incomplete"

        # Ignore agent if necessary (i.e. if comparing controllers for given initial conditions)
        ego = self.ego_id
        if self.ignored_agents is not None:
            self.ignored_agents.sort(reverse=True)
            for ignore in self.ignored_agents:
                z = np.delete(z, ignore, 0)
                if ego > ignore:
                    ego = ego - 1

        # Partition state into ego and other
        ze = z[ego, :]
        zo = np.vstack([z[:ego, :], z[ego + 1 :, :]])

        # Compute nominal control input for ego only -- assume others are zero
        z_copy_nom = z.copy()
        z_copy_nom[self.ego_id] = z[ego]
        u_nom = np.zeros((len(z), self.n_controls))
        u_nom[ego, :], _, _ = self.nominal_controller.compute_control(t, z_copy_nom)
        self.u_nom = u_nom[ego, :]
        self.z_ego = z[ego]

        tuning_nominal = True
        if tuning_nominal:
            self.u = self.u_nom
            return self.u, 1, "Optimal"

        # Partition nominal inputs
        f_nom = u_nom[ego, 0]
        m_nom = u_nom[ego, 1:]

        # Compute force input
        Qf, pf, Af, bf, Gf, hf = self.formulate_force_qp(t, ze, zo, f_nom, ego)
        solf = solve_qp_cvxopt(Qf, pf, Af, bf, Gf, hf)
        if "code" in solf.keys():
            force = solf["x"][0]
            self.force = force
        else:
            status = "Divide by Zero"
            self.u = np.zeros((self.n_controls,))
            return self.u, code, status

        # Compute moment inputs
        Qm, pm, Am, bm, Gm, hm = self.formulate_moment_qp(t, ze, zo, m_nom, ego)
        solm = solve_qp_cvxopt(Qm, pm, Am, bm, Gm, hm)
        if "code" in solm.keys():
            moments = np.array(solm["x"][:3]).flatten()
        else:
            status = "Divide by Zero"
            self.u = np.zeros((self.n_controls,))
            return self.u, code, status

        # moments[0] = 0
        # moments[2] = 0

        u = np.concatenate([[force], moments])
        self.u = np.clip(u, self.u_min, self.u_max)
        code = solf["code"] and solm["code"]
        status = solf["status"] + solm["status"]

        return self.u, code, status

    def formulate_force_qp(
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

        #! Lazy -- figure out better way to do this
        if self.estimator._dt is None:
            self.estimator._dt = self._dt

        # Update Parameter Estimates
        if t > 0:
            # Update estimates
            # self.estimator.update_parameter_estimates(
            #     t, self.z_ego, (self.z_ego - self.z_ego_last) / self._dt
            # )
            # ffunc = self.estimator.compute_unknown_function(self.z_ego, self.u)
            ffunc = np.zeros((12,))

            # # Transmit for nominal control design
            # self.nominal_controller.residual_dynamics = ffunc

            # Logging variables
            residual = f_residual(self.z_ego_last) + g_residual(self.z_ego_last) @ self.u
            self.function_estimation_error = residual - ffunc
            self.function_estimation = ffunc

        # Update last measured state
        self.z_ego_last = self.z_ego

        # Compute Q matrix and p vector for QP objective function
        Q, p = self.compute_objective_qp_force(u_nom)

        # Compute input constraints of form Au @ u <= bu
        Au, bu = self.compute_input_constraints_force()

        # Compute individual CBF constraints
        Ai, bi = self.compute_individual_cbf_constraints_force(ze, ego, condition="naive")

        # Compute pairwise/interagent CBF constraints
        Ap, bp = self.compute_pairwise_cbf_constraints_force(ze, zr, ego)

        A = np.vstack([Au, Ai, Ap])
        b = np.hstack([bu, bi, bp])

        return Q, p, A, b, None, None

    def formulate_moment_qp(
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
        # Compute Q matrix and p vector for QP objective function
        Q, p = self.compute_objective_qp_moments(u_nom)

        # Compute input constraints of form Au @ u <= bu
        Au, bu = self.compute_input_constraints_moments()

        # Compute individual CBF constraints
        Ai, bi = self.compute_individual_cbf_constraints_moments(ze, ego, condition="naive")

        # Compute pairwise/interagent CBF constraints
        Ap, bp = self.compute_pairwise_cbf_constraints_moments(ze, zr, ego)

        A = np.vstack([Au, Ai, Ap])
        b = np.hstack([bu, bi, bp])

        return Q, p, A, b, None, None

    def compute_objective_qp_force(self, u_nom: NDArray) -> (NDArray, NDArray):
        """Computes the matrix Q and vector p for the objective function of the
        form

        J = 1/2 * x.T @ Q @ x + p @ x

        Arguments:
            u_nom: nominal control input for agent in question

        Returns:
            Q: quadratic term positive definite matrix for objective function
            p: linear term vector for objective function

        """
        n_dec_vars = 1
        v_nom = np.concatenate([u_nom.flatten(), np.array(n_dec_vars * [self.nominal_class_k])])
        Q = 1 / 2 * np.array([[1, 0], [0, 100]])
        p = -Q @ v_nom

        return Q, p

    def compute_objective_qp_moments(self, u_nom: NDArray) -> (NDArray, NDArray):
        """Computes the matrix Q and vector p for the objective function of the
        form

        J = 1/2 * x.T @ Q @ x + p @ x

        Arguments:
            u_nom: nominal control input for agent in question

        Returns:
            Q: quadratic term positive definite matrix for objective function
            p: linear term vector for objective function

        """
        n_dec_vars = 3
        Q, p = self.objective(
            np.concatenate([u_nom.flatten(), np.array(n_dec_vars * [self.nominal_class_k])])
        )

        return Q, p

    def compute_input_constraints_force(self):
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
        n_controls = 1
        n_dec_vars = 1
        Au = block_diag(*(self.n_agents * n_controls + n_dec_vars) * [self.au])
        bu = np.append(
            np.array(self.n_agents * [self.bu[: 2 * n_controls]]).flatten(),
            n_dec_vars * [self.max_class_k, 0],
        )

        return Au, bu

    def compute_input_constraints_moments(self):
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
        n_controls = 3
        n_dec_vars = 3
        Au = block_diag(*(self.n_agents * n_controls + n_dec_vars) * [self.au])
        bu = np.append(
            np.array(self.n_agents * [self.bu[2:]]).flatten(),
            n_dec_vars * [self.max_class_k, 0],
        )

        return Au, bu

    def compute_individual_cbf_constraints_force(
        self, ze: NDArray, ego: int, condition: str = "robust"
    ) -> (NDArray, NDArray):
        """Computes matrix Ai and vector bi for individual CBF constraints
        of the form

        Ai @ u <= bi

        where an individual cbf constraint is defined as one for which safety
        is dependent only on the ego agent, not the behavior of other agents
        present in the system.

        Arguments:
            ze: ego state
            ego: identifier of ego agent
            condition: specifies whether to use robust or robust-adaptive condition

        Returns:
            Ai: individual cbf constraint matrix (nCBFs x nControls)
            bi: individual cbf constraint vector (nCBFs x 1)

        """
        # Parameters
        ns = len(ze)  # Number of states
        Ai = np.zeros((1, 2))
        bi = np.zeros((1,))

        if condition == "None":
            return Ai, bi

        # Iterate over individual CBF constraints
        cc = 0
        cbf = self.cbfs_individual[0]
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

        # Robustness bounds
        delta = self.estimator.compute_error_bound(ze)
        bd = delta * np.sum(abs(dhdx))

        if condition == "robust":
            # Get CBF Lie Derivatives
            self.Lfh[cc] = (
                dhdx @ f(ze)
                + dhdx @ self.function_estimation
                - bd
                + stoch
                - self.discretization_error
            )

        elif condition == "robust-adaptive":

            omega_gain = 1e6
            n_params = self.estimator.Px.shape[1]
            Omega = np.eye(n_params) / omega_gain
            del_vec = delta * np.ones((n_params,))
            self.h[cc] -= 1 / 2 * del_vec.T @ Omega @ del_vec
            r_term = (
                np.trace(Omega) * delta * self.estimator.compute_error_bound_derivative(ze) + bd
            )

            # Get CBF Lie Derivatives
            self.Lfh[cc] = (
                dhdx @ f(ze)
                + dhdx @ self.function_estimation
                - r_term
                + stoch
                - self.discretization_error
            )

        elif condition == "standard":
            self.Lfh[cc] = (
                dhdx @ f(ze) + dhdx @ self.function_estimation + stoch - self.discretization_error
            )

        elif condition == "naive":
            self.Lfh[cc] = dhdx @ f(ze) + stoch - self.discretization_error

        # self.Lgh[cc, 0] = dhdx @ g(ze)[:, 0]
        Lgh = np.array([dhdx @ g(ze)[:, 0]])  # [np.newaxis, :]

        # Generate Ai and bi from CBF and associated derivatives
        # Ai[cc], bi[cc] = self.generate_cbf_condition(
        #     cbf, self.h[cc], self.Lfh[cc], self.Lgh[cc], cc, adaptive=True
        # )
        Ai[cc], bi[cc] = self.generate_cbf_condition(
            cbf, self.h[cc], self.Lfh[cc], Lgh, cc, adaptive=True
        )

        # Check whether any of the safety conditions are violated
        if (np.sum(self.h0[cc] < 0) > 0) or (np.sum(self.h[cc] < 0) > 0):
            self.safety = False
            print(f"Safety Violation!! h0 --> {self.h0[cc]}")
            print(f"Safety Violation!! h  --> {self.h[cc]}")

        return Ai, bi

    def compute_individual_cbf_constraints_moments(
        self, ze: NDArray, ego: int, condition: str = "robust"
    ) -> (NDArray, NDArray):
        """Computes matrix Ai and vector bi for individual CBF constraints
        of the form

        Ai @ u <= bi

        where an individual cbf constraint is defined as one for which safety
        is dependent only on the ego agent, not the behavior of other agents
        present in the system.

        Arguments:
            ze: ego state
            ego: identifier of ego agent
            condition: specifies whether to use robust or robust-adaptive condition

        Returns:
            Ai: individual cbf constraint matrix (nCBFs x nControls)
            bi: individual cbf constraint vector (nCBFs x 1)

        """
        # Parameters
        ns = len(ze)  # Number of states
        lmi = len(self.cbfs_individual) - 1
        n_controls = 3
        Ai = np.zeros((lmi, 3 * self.n_agents + 3))
        bi = np.zeros((lmi,))

        if condition == "None":
            return Ai, bi

        # Iterate over individual CBF constraints
        for cc, cbf in enumerate(self.cbfs_individual[1:]):
            self.h0[cc] = cbf.h0(ze, self.force)
            self.h[cc] = cbf.h(ze, self.force)
            self.dhdx[cc][:ns] = cbf.dhdx(ze, self.force)
            dhdx = self.dhdx[cc][:ns]

            # Stochastic Term -- 0 for deterministic systems
            if np.trace(sigma(ze).T @ sigma(ze)) > 0 and self._stochastic:
                d2hdx2 = cbf.d2hdx2(ze)
                stoch = 0.5 * np.trace(sigma(ze).T @ d2hdx2 @ sigma(ze))
            else:
                stoch = 0.0

            # Robustness bounds
            delta = self.estimator.compute_error_bound(ze)
            bd = delta * np.sum(abs(dhdx))

            if condition == "robust":
                # Get CBF Lie Derivatives
                self.Lfh[cc] = (
                    dhdx @ f(ze)
                    + dhdx @ self.function_estimation
                    - bd
                    + stoch
                    - self.discretization_error
                )

            elif condition == "robust-adaptive":

                omega_gain = 1e6
                n_params = self.estimator.Px.shape[1]
                Omega = np.eye(n_params) / omega_gain
                del_vec = delta * np.ones((n_params,))
                self.h[cc] -= 1 / 2 * del_vec.T @ Omega @ del_vec
                r_term = (
                    np.trace(Omega) * delta * self.estimator.compute_error_bound_derivative(ze) + bd
                )

                # Get CBF Lie Derivatives
                self.Lfh[cc] = (
                    dhdx @ f(ze)
                    + dhdx @ self.function_estimation
                    - r_term
                    + stoch
                    - self.discretization_error
                )

            elif condition == "standard":
                self.Lfh[cc] = (
                    dhdx @ f(ze)
                    + dhdx @ self.function_estimation
                    + stoch
                    - self.discretization_error
                )

            elif condition == "naive":
                self.Lfh[cc] = dhdx @ f(ze) + stoch - self.discretization_error

            self.Lgh[cc, n_controls * ego : (ego + 1) * n_controls] = (
                dhdx @ g(ze)[:, 1:]
            )  # Only assign ego control

            # Generate Ai and bi from CBF and associated derivatives
            adaptive = cc == 2
            A_temp, bi[cc] = self.generate_cbf_condition(
                cbf, self.h[cc], self.Lfh[cc], self.Lgh[cc, 1:], cc, adaptive=adaptive
            )

            # Parse returns from generate condition
            Ai[cc, : n_controls * self.n_agents] = A_temp[:-1]  # Control terms
            Ai[cc, n_controls * self.n_agents + cc] = A_temp[-1]  # Alpha term

        # Check whether any of the safety conditions are violated
        if (np.sum(self.h0[1 : len(self.cbfs_individual)] < 0) > 0) or (
            np.sum(self.h[1 : len(self.cbfs_individual)] < 0) > 0
        ):
            self.safety = False
            print(f"Safety Violation!! h0 --> {np.min(self.h0[1:len(self.cbfs_individual)])}")
            print(f"Safety Violation!! h  --> {np.min(self.h[1:len(self.cbfs_individual)])}")

        return Ai, bi

    def compute_pairwise_cbf_constraints_force(
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
        Ap = np.zeros((len(self.cbfs_pairwise) * len(zr), 1 * self.n_agents + 1))
        bp = np.zeros((len(self.cbfs_pairwise) * len(zr),))

        return Ap, bp

        # # Iterate over pairwise CBF constraints
        # for cc, cbf in enumerate(self.cbfs_pairwise):

        #     # Iterate over all other vehicles
        #     for ii, zo in enumerate(zr):
        #         other = ii + (ii >= ego)
        #         idx = lci + cc * zr.shape[0] + ii

        #         self.h0[idx] = cbf.h0(ze, zo)
        #         self.h[idx] = cbf.h(ze, zo)
        #         self.dhdx[idx] = cbf.dhdx(ze, zo)
        #         dhdx = self.dhdx[idx]

        #         # Stochastic Term -- 0 for deterministic systems
        #         if np.trace(sigma(ze).T @ sigma(ze)) > 0 and self._stochastic:
        #             d2hdx2 = cbf.d2hdx2(ze, zo)
        #             stoch = 0.5 * (
        #                 np.trace(sigma(ze).T @ d2hdx2[:ns, :ns] @ sigma(ze))
        #                 + np.trace(sigma(zo).T @ d2hdx2[ns:, ns:] @ sigma(zo))
        #             )
        #         else:
        #             stoch = 0.0

        #         # Time-Varying Parameter Term
        #         tv_term = self.compute_time_varying_cbf_term(dhdx)

        #         # Get CBF Lie Derivatives
        #         self.Lfh[idx] = (
        #             dhdx[:ns] @ f(ze)
        #             + dhdx[ns:] @ f(zo)
        #             + stoch
        #             - self.discretization_error
        #             + tv_term
        #         )
        #         self.Lgh[idx, self.n_controls * ego : (ego + 1) * self.n_controls] = dhdx[:ns] @ g(
        #             ze
        #         )
        #         self.Lgh[idx, self.n_controls * other : (other + 1) * self.n_controls] = dhdx[
        #             ns:
        #         ] @ g(zo)

        #         # Generate Ap and bp from CBF and associated derivatives
        #         p_idx = cc * len(self.cbfs_pairwise) + ii
        #         A_temp, bp[p_idx] = self.generate_cbf_condition(
        #             cbf, self.h[idx], self.Lfh[idx], self.Lgh[idx], idx, adaptive=True
        #         )

        #         Ap[p_idx, : self.n_controls * self.n_agents] = A_temp[:-1]  # Control terms
        #         Ap[p_idx, self.n_controls * self.n_agents + p_idx] = A_temp[-1]  # Alpha term

        #         # Check whether any of the safety conditions are violated
        # if (np.sum(self.h0[len(self.cbfs_individual) :] < 0) > 0) or (
        #     np.sum(self.h[len(self.cbfs_individual) :] < 0) > 0
        # ):
        #     self.safety = False
        #     print(f"Safety Violation!! h0 --> {np.min(self.h0[len(self.cbfs_individual):])}")
        #     print(f"Safety Violation!! h  --> {np.min(self.h[len(self.cbfs_individual):])}")

        # return Ap, bp

    def compute_pairwise_cbf_constraints_moments(
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
        Ap = np.zeros((len(self.cbfs_pairwise) * len(zr), 3 * self.n_agents + 3))
        bp = np.zeros((len(self.cbfs_pairwise) * len(zr),))

        return Ap, bp

        # # Iterate over pairwise CBF constraints
        # for cc, cbf in enumerate(self.cbfs_pairwise):

        #     # Iterate over all other vehicles
        #     for ii, zo in enumerate(zr):
        #         other = ii + (ii >= ego)
        #         idx = lci + cc * zr.shape[0] + ii

        #         self.h0[idx] = cbf.h0(ze, zo)
        #         self.h[idx] = cbf.h(ze, zo)
        #         self.dhdx[idx] = cbf.dhdx(ze, zo)
        #         dhdx = self.dhdx[idx]

        #         # Stochastic Term -- 0 for deterministic systems
        #         if np.trace(sigma(ze).T @ sigma(ze)) > 0 and self._stochastic:
        #             d2hdx2 = cbf.d2hdx2(ze, zo)
        #             stoch = 0.5 * (
        #                 np.trace(sigma(ze).T @ d2hdx2[:ns, :ns] @ sigma(ze))
        #                 + np.trace(sigma(zo).T @ d2hdx2[ns:, ns:] @ sigma(zo))
        #             )
        #         else:
        #             stoch = 0.0

        #         # Time-Varying Parameter Term
        #         tv_term = self.compute_time_varying_cbf_term(dhdx)

        #         # Get CBF Lie Derivatives
        #         self.Lfh[idx] = (
        #             dhdx[:ns] @ f(ze)
        #             + dhdx[ns:] @ f(zo)
        #             + stoch
        #             - self.discretization_error
        #             + tv_term
        #         )
        #         self.Lgh[idx, self.n_controls * ego : (ego + 1) * self.n_controls] = dhdx[:ns] @ g(
        #             ze
        #         )
        #         self.Lgh[idx, self.n_controls * other : (other + 1) * self.n_controls] = dhdx[
        #             ns:
        #         ] @ g(zo)

        #         # Generate Ap and bp from CBF and associated derivatives
        #         p_idx = cc * len(self.cbfs_pairwise) + ii
        #         A_temp, bp[p_idx] = self.generate_cbf_condition(
        #             cbf, self.h[idx], self.Lfh[idx], self.Lgh[idx], idx, adaptive=True
        #         )

        #         Ap[p_idx, : self.n_controls * self.n_agents] = A_temp[:-1]  # Control terms
        #         Ap[p_idx, self.n_controls * self.n_agents + p_idx] = A_temp[-1]  # Alpha term

        #         # Check whether any of the safety conditions are violated
        # if (np.sum(self.h0[len(self.cbfs_individual) :] < 0) > 0) or (
        #     np.sum(self.h[len(self.cbfs_individual) :] < 0) > 0
        # ):
        #     self.safety = False
        #     print(f"Safety Violation!! h0 --> {np.min(self.h0[len(self.cbfs_individual):])}")
        #     print(f"Safety Violation!! h  --> {np.min(self.h[len(self.cbfs_individual):])}")

        # return Ap, bp

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


#! This will eventually go in another module
