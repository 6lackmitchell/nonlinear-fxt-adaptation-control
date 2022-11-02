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
from scipy.linalg import block_diag, null_space, logm
from collections import deque

# from core.cbfs.cbf import Cbf
from core.controllers.cbf_qp_controller import CbfQpController
from core.controllers.controller import Controller
from core.mathematics.basis_functions import basis_functions, basis_function_gradients

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


def sigmoid(x: float or NDArray):
    """Numerically stable implementation of the sigmoid function.

    Arguments:
        x: input

    Returns:
        sigmoid(x)

    """
    sig = np.where(x < 0, np.exp(x) / (1 + np.exp(x)), 1 / (1 + np.exp(-x)))

    return sig


#! This will eventually go in another module
class RecurrentNeuralNetwork:
    """RecurrentNeuralNetwork: class interface to recurrent neural network.

    Properties:
        TBD

    Methods:
        TBD

    """

    def __init__(self, n_inputs: int, n_outputs: int):
        """Class initializer.

        Arguments:
            n_inputs: number of RNN inputs
            n_outputs: number of RNN outputs

        """
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        # Weights
        self.input_weights_a = 0.1 * np.eye(n_inputs)  # Forget Weights
        self.input_weights_b = np.eye(n_inputs)
        self.input_weights_c = np.eye(n_inputs)
        self.input_weights_d = np.eye(n_inputs)
        self.hidden_weights_a = 0.1 * np.eye(n_outputs)  # Forget Weights
        self.hidden_weights_b = np.eye(n_outputs)
        self.hidden_weights_c = np.eye(n_outputs)
        self.hidden_weights_d = np.eye(n_outputs)

        # Biases
        self.bias_a = np.random.random((n_inputs,))
        self.bias_b = np.random.random((n_inputs,))
        self.bias_c = np.random.random((n_inputs,))
        self.bias_d = np.random.random((n_inputs,))

        # RNN States
        self.cell_state = np.zeros((self.n_inputs,))
        self.hidden_state = np.zeros((self.n_outputs,))

    #! This is really for a LSTM RNN, will generalize later
    def update_rnn(self, new_input: NDArray) -> NDArray:
        """Updates the RNN state according to the new input.

        Arguments:
            new_input: new input to the RNN

        Returns:
            new_output: new output based on input, hidden, and cell states

        """
        at = sigmoid(
            self.hidden_weights_a @ self.hidden_state
            + self.input_weights_a @ new_input
            + self.bias_a
        )
        bt = sigmoid(
            self.hidden_weights_b @ self.hidden_state
            + self.input_weights_b @ new_input
            + self.bias_b
        )
        ct = np.tanh(
            self.hidden_weights_c @ self.hidden_state
            + self.input_weights_c @ new_input
            + self.bias_c
        )
        dt = sigmoid(
            self.hidden_weights_d @ self.hidden_state
            + self.input_weights_d @ new_input
            + self.bias_d
        )

        self.cell_state = self.hidden_state * at + bt * ct
        self.hidden_state = dt * np.tanh(self.cell_state)

        return self.hidden_state

    @property
    def outputs(self) -> NDArray:
        """Property for the hidden RNN states."""
        return self.hidden_state


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
        nStates: int,
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
        self.n_agents = nAgents
        self.n_states = nStates

        # Unknown parameter properties
        example_basis_fcns = basis_functions(np.zeros((self.n_states,)))
        self.n_params = len(example_basis_fcns) ** 2
        self.theta = np.eye(len(example_basis_fcns)).reshape((self.n_params,))
        self.theta_dot = np.zeros((self.n_params,))
        self.eta0 = 100.0 * np.ones((self.n_params,))  #! Lazy coding -- needs to be fixed
        self.M = np.zeros((len(example_basis_fcns), self.n_params))
        self.Mf = np.zeros((len(example_basis_fcns), self.n_states))
        self.ffunc = np.zeros((self.n_states,))
        self.ffunc_dot = np.zeros((self.n_states,))
        self.U_koopman = np.eye(len(example_basis_fcns))
        self.U_koopman_last = np.eye(len(example_basis_fcns))
        self.L_generator = np.zeros((len(example_basis_fcns), len(example_basis_fcns)))
        self.function_estimation_error = np.zeros((self.n_states,))
        self.function_estimation = np.zeros((self.n_states,))

        # Deques for testing least squares approach
        self.PX = deque([], maxlen=100)
        self.PY = deque([], maxlen=100)
        self.DPXDX = deque([], maxlen=100)

        # RNN's for Basis Function Memory
        self.rnn_px = RecurrentNeuralNetwork(len(example_basis_fcns), len(example_basis_fcns))
        self.rnn_py = RecurrentNeuralNetwork(len(example_basis_fcns), len(example_basis_fcns))
        self.rnn_dpxdx = RecurrentNeuralNetwork(
            len(example_basis_fcns) * self.n_states, len(example_basis_fcns) * self.n_states
        )

        # Gains -- a = 0 becomes finite-time
        # self.law_gains = {"a": 1.0, "b": 1.0, "w": 5.0, "G": 1e-3 * np.eye(self.n_params)}
        self.law_gains = {"a": 5.0, "b": 5.0, "w": 5.0, "G": 1e-1 * np.eye(self.n_params)}

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
            # Update estimates
            theta, theta_dot = self.update_parameter_estimates()
            # ffunc, ffunc_dot = self.update_unknown_function_estimate()
            # ffunc = self.estimate_uncertainty_lstsq()
            ffunc = self.compute_unknown_function()

            # residual_dynamics = self.compute_unknown_function()
            function_estimation_error = f_residual(self.z_ego) - ffunc
            self.nominal_controller.residual_dynamics = ffunc

            self.function_estimation_error = function_estimation_error
            self.function_estimation = ffunc
            # print(f"Residual Dynamics: {residual_dynamics}")

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
            Au = block_diag(*(self.n_agents * self.n_controls + self.n_dec_vars) * [self.au])
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
        self.theta[abs(self.theta) < 1e-6] = 0  # Step used in Mauroy et al.

        return self.theta, self.theta_dot

    def estimate_uncertainty_lstsq(self) -> NDArray:
        """Tests the Koopman approximation approach using the standard data-
        driven least squares method.

        Arguments:
            None

        Returns:
            theta: updated parameter estimates for approximating the uncertainty
            theta_dot: array of zeros -- not used in data-driven approach

        """
        # Compute lifted data in basis space
        self.PX.append(self.compute_basis_functions(self.z_ego))
        self.PY.append(self.compute_basis_functions(self.outputs))
        self.DPXDX.append(self.compute_basis_function_gradients(self.z_ego))

        PX = np.array(self.PX)
        PY = np.array(self.PY)
        DPXDX = np.vstack(self.DPXDX)

        # Approximate Koopman matrix with least squares
        U = np.linalg.pinv(PX) @ PY
        rank_U = np.linalg.matrix_rank(U)
        min_eig_U = np.min(np.linalg.eig(U)[0])

        # If U is singular or has any negative real eigenvalues, then logm(U) is undefined
        if rank_U < U.shape[0]:
            return np.zeros((self.n_states,))

        # Approximate Koopman generator
        A = 1 / self._dt * logm(U)

        # Approximate Vector Field
        F = np.linalg.pinv(DPXDX) @ (block_diag(*(PX.shape[0]) * [A.T])) @ PX.flatten()

        return F

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
        # px = self.rnn_px.update_rnn(self.compute_basis_functions(self.z_ego))
        # py = self.rnn_py.update_rnn(self.compute_basis_functions(self.outputs))
        px = self.compute_basis_functions(self.z_ego)
        py = self.compute_basis_functions(self.outputs)

        # Compute matrix M and vector v for adaptation law
        self.M = block_diag(*(len(px)) * [px])
        v = py - self.M @ self.theta

        # Load gains
        a = self.law_gains["a"]
        b = self.law_gains["b"]
        w = self.law_gains["w"]
        G = self.law_gains["G"]
        norm_v = np.linalg.norm(v)

        # Compute adaptation
        theta_dot = G @ self.M.T @ v * (a * norm_v ** (2 / w) + b / norm_v ** (2 / w))

        # Check for unbounded parameters
        if np.max(self.theta) > 1e9:
            print(self.theta[self.theta > 1e9])

        return theta_dot

    def update_unknown_function_estimate(self) -> NDArray:
        """Updates the estimated unknown function in the system dynamics
        according to the FxTS update law.

        Arguments:
            TBD

        Returns:
            TBD

        """
        # Compute time-derivatives of unknown function
        self.ffunc_dot = self.compute_ffunc_dot()

        # Update theta parameters according to first-order forward-Euler
        self.ffunc = self.ffunc + self.ffunc_dot * self._dt
        self.ffunc[abs(self.ffunc) < 1e-6] = 0  # Step used in Mauroy et al.

        return self.ffunc, self.ffunc_dot

    def compute_ffunc_dot(self) -> NDArray:
        """Computes the time-derivative of the estimated unknown function
        in the system dynamics.

        Arguments:
            None

        Returns:
            ffunc_dot: time-derivative of unknown function estimate

        """
        # Generate psi and partial derivatives
        # px = self.rnn_px.outputs
        # gradient_matrix = self.compute_basis_function_gradients(self.z_ego)
        # dpxdx = self.rnn_dpxdx.update_rnn(
        #     gradient_matrix.reshape((len(px) * self.n_states,))
        # ).reshape(gradient_matrix.shape)
        px = self.compute_basis_functions(self.z_ego)
        dpxdx = self.compute_basis_function_gradients(self.z_ego)

        self.Mf = dpxdx
        v = self.compute_koopman_generator().T @ px - self.Mf @ self.ffunc
        # print(f"v: {v}")
        # print(f"L: {self.compute_koopman_generator().T.max()}")
        # print(f"p: {px.max()}")
        # print(f"M: {self.Mf.max()}")
        # print(f"f: {self.ffunc.max()}")

        a = self.law_gains["a"]
        b = self.law_gains["b"]
        w = self.law_gains["w"]
        G = self.law_gains["G"]
        norm_v = np.linalg.norm(v)

        if norm_v > 1e-6:
            ffunc_dot = (
                G[: self.n_states, : self.n_states]
                @ self.Mf.T
                @ v
                * (a * norm_v ** (2 / w) + b / norm_v ** (2 / w))
            )
        else:
            ffunc_dot = np.zeros(self.ffunc_dot.shape)

        return ffunc_dot

    def compute_unknown_function(self) -> NDArray:
        """Computes the approximate infinitesimal generator L of the
        Koopman Operator U.

        Arguments
            TBD

        Returns
            unknown_f_estimate: estimated unknown nonlinear function

        """
        z = self.z_ego
        # unknown_f_estimate = (
        #     self.compute_basis_functions(z) @ self.compute_koopman_generator()[:, : self.n_states]
        # )

        # # Get RNN Data
        # px = self.rnn_px.outputs
        # gradient_matrix = self.compute_basis_function_gradients(self.z_ego)
        # dpxdx = self.rnn_dpxdx.update_rnn(
        #     gradient_matrix.reshape((len(px) * self.n_states,))
        # ).reshape(gradient_matrix.shape)

        # Approximate Vector Field
        px = self.compute_basis_functions(self.z_ego)
        dpxdx = self.compute_basis_function_gradients(self.z_ego)
        unknown_f_estimate = np.linalg.pinv(dpxdx) @ (self.compute_koopman_generator().T @ px)

        # unknown_f_estimate = (
        #     self.rnn_px.outputs @ self.compute_koopman_generator()[:, : self.n_states]
        # )

        return unknown_f_estimate

    def compute_basis_functions(self, z: NDArray) -> NDArray:
        """Computes the values of the basis functions evaluated at the current
        state and control values. Returns the (b x 1) vector of basis function
        values.

        Arguments
            z: input vector (may be states and controls or outputs)

        Returns
            basis_functions: vector of values of basis functions

        """
        return basis_functions(z)

    def compute_basis_function_gradients(self, z: NDArray) -> NDArray:
        """Computes the gradients of the basis functions evaluated at the current
        state and control values. Returns the (b x n) matrix of basis function
        gradients.

        Arguments
            z: input vector (may be states and controls or outputs)

        Returns
            basis_function_grads: matrix of gradients of basis functions

        """
        return basis_function_gradients(z)

    def get_koopman_matrix(self) -> NDArray:
        """Retrieves the (approximated) Koopman matrix from the parameter
        estimates theta.

        Arguments
            None

        Returns
            U: (approximate) Koopman operator

        """
        dim_U = self.M.shape[0]
        self.U_koopman = self.theta.reshape((dim_U, dim_U)).T

        return self.U_koopman

    def compute_koopman_generator(self) -> NDArray:
        """Computes the approximate infinitesimal generator L of the
        Koopman Operator U.

        Arguments
            TBD

        Returns
            L: (approximate) infinitesimal generator of Koopman operator

        """
        U = self.get_koopman_matrix()
        rank_U = np.linalg.matrix_rank(U)
        min_eig_U = np.min(np.linalg.eig(U)[0])

        # If U is singular or has any negative real eigenvalues, then logm(U) is undefined
        if rank_U < U.shape[0]:  # or min_eig_U < 0:
            raise ValueError("Linearly Dependent Koopman Matrix --> No LogM Generator!")
        else:
            # Discrete-Sampling Implementation
            self.L_generator = logm(U) / self._dt

            # # Numerically Differentiate (Continuous-Time Approximation)
            # self.L_generator = (logm(U) - logm(self.U_koopman_last)) / self._dt
            # self.U_koopman_last = U

        # *******************
        self.L_generator[abs(self.L_generator) < 1e-6] = 0
        # This step was introduced by Mauroy et al.
        # in line 150 of (https://github.com/AlexMauroy/Koopman-identification/blob/master/main/matlab/lifting_ident_main.m)
        # *******************

        return self.L_generator

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

    @property
    def outputs(self):
        """Property for computing 'output' for estimator. In this case, the
        estimator seeks to estimate the unknown residual dynamics in the
        system."""

        xdot = (self.z_ego - self.z_ego_last) / self._dt
        residual = xdot - f(self.z_ego) - g(self.z_ego) @ self.u

        outputs = self.z_ego + xdot * self._dt

        return outputs
