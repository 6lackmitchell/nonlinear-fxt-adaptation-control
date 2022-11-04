"""koopman_estimators.py

Contains classes to be used for Koopman Operator based estimation.

Classes:
    KoopmanEstimator (Parent)
    KoopmanMatrixEstimator
    KoopmanGeneratorEstimator
    DataDrivenKoopmanMatrixEstimator
    DataDrivenKoopmanGeneratorEstimator

"""
from nptyping import NDArray
from scipy.linalg import block_diag, logm
import numpy as np
from collections import deque
from core.mathematics.basis_functions import basis_functions, basis_function_gradients
from core.networks.recurrent_neural_network import RecurrentNeuralNetwork

# MONOMIAL_FACTOR = 100.0
MONOMIAL_FACTOR = 1.0


class KoopmanEstimator:
    """Parent class for Koopman operator based estimators. Not designed for
    stand-alone use.

    Attributes:
        n_states
        n_params
        koopman_matrix
        koopman_generator
        theta

    Methods:
        update_parameter_estimates
        compute_theta_dot
        compute_basis_functions
        compute_basis_function_gradients

    """

    def __init__(
        self,
        n_states: int,
        dt: float,
        nominal_f: callable,
        nominal_g: callable,
        use_rnn: bool = False,
    ):
        """Class initializer.

        Arguments:
            n_states: number of states (i.e. dimension of vector field)
            dt: simulation timestep

        """
        self._dt = dt
        self.use_rnn = use_rnn
        self.n_states = n_states
        self.nominal_f = nominal_f
        self.nominal_g = nominal_g
        self.n_params = len(basis_functions(np.zeros((self.n_states,))))
        self.xdot_meas = np.zeros((n_states,))

        # Initialize Koopman objects
        self.koopman_matrix = np.zeros((self.n_params, self.n_params))
        self.koopman_generator = np.zeros((self.n_params, self.n_params))

        # Initialize parameter estimates
        # self.theta = np.zeros((self.n_params**2,))
        self.theta = np.eye(self.n_params).reshape((self.n_params**2,))

        # Initialize lifted input/output matrix/vectors
        self.Px = np.zeros((self.n_params, self.n_params**2))
        self.Py = np.zeros((self.n_params,))

        # Define RNNs for Basis Function Memory
        self.rnn_px = RecurrentNeuralNetwork(self.n_params, self.n_params)
        self.rnn_py = RecurrentNeuralNetwork(self.n_params, self.n_params)
        self.rnn_dpxdx = RecurrentNeuralNetwork(
            self.n_params * self.n_states, self.n_params * self.n_states
        )

        # *******************
        # Adaptation gains
        a = 1.0
        b = 1.0
        w = 5.0

        G = 25  # Sinusoid Example Generator Estimator -- sin bases
        # G = 1  # Sinusoid Example Matrix Estimator -- sin bases
        # G = 25  # Rossler Example Generator Estimator
        # G = 1  # Rossler Example Matrix Estimator
        # G = 1  # Quadrotor Example Generator Estimator -- monomial bases
        # G = 1  # Quadrotor Example Matrix Estimator -- monomial bases

        # G = 1e-1

        # Define adaptation gains
        self.law_gains = {
            "a": a,
            "b": b,
            "w": w,
            "G": G * np.eye(self.n_params**2),
        }

        # Backup gains
        # self.law_gains = {
        #     "a": 1.0,
        #     "b": 1.0,
        #     "w": 5.0,
        #     "G": 1 * np.eye(self.n_params),
        # }  # Works for sin/cos sinusoidal basis functions
        # self.law_gains = {
        #     "a": 1.0,
        #     "b": 1.0,
        #     "w": 5.0,
        #     "G": 10 * np.eye(self.n_params),
        # }  # Works for Monomial and sinusoidal basis functions
        # self.law_gains = {
        #     "a": 1.0,
        #     "b": 1.0,
        #     "w": 5.0,
        #     "G": 1e-1 * np.eye(self.n_params),
        # }  # Testing for quadrotor -- okay for monomials
        # self.law_gains = {
        #     "a": 1.0,
        #     "b": 1.0,
        #     "w": 5.0,
        #     "G": 1e-3 * np.eye(self.n_params),
        # }  # Testing for quadrotor

    def update_parameter_estimates(self, z: NDArray, xdot_meas: NDArray) -> NDArray:
        """Updates parameters comprising the approximated Koopman Operator
        according to the following parameter update law:

        thetadot = Gamma @ M.T @ v * (a * ||v||^(2 / u) + b / ||v||^(2 / u))

        where M and v are related according to Mz = v, with z the parameter
        estimation error.

        Arguments:
            z: state vector
            xdot_meas: last measured value of xdot

        Returns:
            theta: updated parameter estimates

        """
        self.xdot_meas = xdot_meas

        # Compute time-derivatives of theta parameters
        theta_dot = self.compute_theta_dot(z)

        # Update theta parameters according to first-order forward-Euler
        self.theta = self.theta + theta_dot * self._dt
        self.theta[abs(self.theta) < 1e-12] = 0  # Step used in Mauroy et al.

        return self.theta

    def compute_theta_dot(self, z: NDArray) -> NDArray:
        """Computes the time-derivative of the Koopman matrix parameters according
        to the following parameter update law:

        thetadot = Gamma @ M.T @ v * (a * ||v||^(2 / u) + b / ||v||^(2 / u))

        where M and v are related according to Mz = v, with z the parameter
        estimation error.

        Arguments:
            z: state vector

        Returns:
            theta_dot: time-derivative of parameter estimates in system dynamics

        """
        # Generate Px and Py from input/output data
        if self.use_rnn:
            # Update RNN states (including dpxdx which is not used here)
            px = self.rnn_px.update_rnn(self.compute_lifted_inputs(z))
            py = self.rnn_py.update_rnn(self.compute_lifted_outputs(z))
            gradient_matrix = self.compute_basis_function_gradients(z)
            dpxdx = self.rnn_dpxdx.update_rnn(
                gradient_matrix.reshape((len(px) * self.n_states,))
            ).reshape(gradient_matrix.shape)
        else:
            px = self.compute_lifted_inputs(z)
            py = self.compute_lifted_outputs(z)

        # Compute matrix M and vector v for adaptation law
        self.Px = block_diag(*(len(px)) * [px])
        v = py - self.Px @ self.theta

        # Load gains
        a = self.law_gains["a"]
        b = self.law_gains["b"]
        w = self.law_gains["w"]
        G = self.law_gains["G"]
        norm_v = np.linalg.norm(v)

        # Compute adaptation
        theta_dot = G @ self.Px.T @ v * (a * norm_v ** (2 / w) + b / norm_v ** (2 / w))

        return theta_dot

    def compute_unknown_function(self, z: NDArray, u: NDArray) -> NDArray:
        """Computes the approximate infinitesimal generator L of the
        Koopman Operator U.

        Arguments
            z: state vector
            u: control input vector

        Returns
            unknown_residual_estimate: estimated unknown nonlinear function

        """
        if self.use_rnn:
            px = self.rnn_px.outputs
            dpxdx = self.rnn_dpxdx.outputs.reshape((self.n_params, self.n_states))

        else:
            px = self.compute_basis_functions(z)
            dpxdx = self.compute_basis_function_gradients(z)

        # Estimate total vector field xdot = F(x)
        total_vector_field_estimate = (
            np.linalg.pinv(dpxdx) @ (self.get_koopman_generator().T @ px)
        ) * MONOMIAL_FACTOR

        unknown_residual_estimate = total_vector_field_estimate - (
            self.nominal_f(z) + self.nominal_g(z) @ u
        )

        return unknown_residual_estimate

    def compute_basis_functions(self, z: NDArray) -> NDArray:
        """Computes the values of the basis functions evaluated at the current
        state and control values. Returns the (b x 1) vector of basis function
        values.

        Arguments
            z: input vector (may be states and controls or outputs)

        Returns
            basis_functions: vector of values of basis functions

        """
        return basis_functions(z / MONOMIAL_FACTOR)

    def compute_basis_function_gradients(self, z: NDArray) -> NDArray:
        """Computes the gradients of the basis functions evaluated at the current
        state and control values. Returns the (b x n) matrix of basis function
        gradients.

        Arguments
            z: input vector (may be states and controls or outputs)

        Returns
            basis_function_grads: matrix of gradients of basis functions

        """
        return basis_function_gradients(z / MONOMIAL_FACTOR)

    def compute_lifted_inputs(self, z: NDArray) -> NDArray:
        """Computes the Koopman inputs lifted to the basis space.

        Arguments:
            z: input vector (usually state vector)

        Returns:
            Px: lifted inputs in basis space

        """
        return self.compute_basis_functions(z)

    def compute_lifted_outputs(self, z: NDArray) -> NDArray:
        """Gets the lifted basis outputs for the Koopman estimator according
        to the child estimation scheme.

        Arguments:
            z: state vector
            u: control input vector

        Returns:
            outputs: vector of lifted basis outputs

        """
        return self._compute_lifted_outputs(z)  # Overloaded by child class

    def get_koopman_generator(self) -> NDArray:
        """Gets the Koopman generator according to the child estimation scheme.

        Arguments:
            None

        Returns:
            koopman_generator: vector of lifted basis outputs

        """
        return self._get_koopman_generator()  # Overloaded by child class

    # # For cascading FxT estimation -- not used right now
    # def update_unknown_function_estimate(self) -> NDArray:
    #     """Updates the estimated unknown function in the system dynamics
    #     according to the FxTS update law.

    #     Arguments:
    #         TBD

    #     Returns:
    #         TBD

    #     """
    #     # Compute time-derivatives of unknown function
    #     self.ffunc_dot = self.compute_ffunc_dot()

    #     # Update theta parameters according to first-order forward-Euler
    #     self.ffunc = self.ffunc + self.ffunc_dot * self._dt
    #     self.ffunc[abs(self.ffunc) < 1e-12] = 0  # Step used in Mauroy et al.

    #     return self.ffunc, self.ffunc_dot

    # def compute_ffunc_dot(self) -> NDArray:
    #     """Computes the time-derivative of the estimated unknown function
    #     in the system dynamics.

    #     Arguments:
    #         None

    #     Returns:
    #         ffunc_dot: time-derivative of unknown function estimate

    #     """
    #     # Generate psi and partial derivatives
    #     # px = self.rnn_px.outputs
    #     # gradient_matrix = self.compute_basis_function_gradients(self.z_ego)
    #     # dpxdx = self.rnn_dpxdx.update_rnn(
    #     #     gradient_matrix.reshape((len(px) * self.n_states,))
    #     # ).reshape(gradient_matrix.shape)
    #     px = self.compute_basis_functions(self.z_ego)
    #     dpxdx = self.compute_basis_function_gradients(self.z_ego)

    #     self.Mf = dpxdx
    #     v = self.compute_koopman_generator().T @ px - self.Mf @ self.ffunc
    #     # print(f"v: {v}")
    #     # print(f"L: {self.compute_koopman_generator().T.max()}")
    #     # print(f"p: {px.max()}")
    #     # print(f"M: {self.Mf.max()}")
    #     # print(f"f: {self.ffunc.max()}")

    #     a = self.law_gains["a"]
    #     b = self.law_gains["b"]
    #     w = self.law_gains["w"]
    #     G = self.law_gains["G"]
    #     norm_v = np.linalg.norm(v)

    #     if norm_v > 1e-6:
    #         ffunc_dot = (
    #             G[: self.n_states, : self.n_states]
    #             @ self.Mf.T
    #             @ v
    #             * (a * norm_v ** (2 / w) + b / norm_v ** (2 / w))
    #         )
    #     else:
    #         ffunc_dot = np.zeros(self.ffunc_dot.shape)

    #     return ffunc_dot


class KoopmanMatrixEstimator(KoopmanEstimator):
    """Interface to the parameter adaptation law estimating the Koopman matrix.

    Methods:
        TBD

    """

    def __init__(
        self,
        n_states: int,
        dt: float,
        nominal_f: callable,
        nominal_g: callable,
        use_rnn: bool = False,
    ):
        """Class initializer.

        Arguments:
            n_states: number of states (i.e. dimension of vector field)

        """
        super().__init__(n_states, dt, nominal_f, nominal_g, use_rnn)

    def get_koopman_matrix(self) -> NDArray:
        """Obtains the Koopman matrix from the adapted parameters.

        Arguments:
            None

        Returns:
            koopman_matrix

        """
        self.koopman_matrix = self.theta.reshape((self.n_params, self.n_params)).T

        return self.koopman_matrix

    def _get_koopman_generator(self) -> NDArray:
        """Computes the approximate infinitesimal generator L of the
        Koopman Operator U.

        Arguments
            TBD

        Returns
            koopman_generator: (approximate) infinitesimal generator of Koopman operator

        """
        U = self.get_koopman_matrix()
        rank_U = np.linalg.matrix_rank(U)
        min_eig_U = np.min(np.linalg.eig(U)[0])

        # If U is singular or has any negative real eigenvalues, then logm(U) is undefined
        if rank_U < U.shape[0]:  # or min_eig_U < 0:
            raise ValueError("Linearly Dependent Koopman Matrix --> No LogM Generator!")
        else:
            # Discrete-Sampling Implementation
            self.koopman_generator = logm(U) / self._dt

        # *******************
        # This step was introduced by Mauroy et al.
        # in line 150 of (https://github.com/AlexMauroy/Koopman-identification/blob/master/main/matlab/lifting_ident_main.m)
        self.koopman_generator[abs(self.koopman_generator) < 1e-12] = 0
        # *******************

        return self.koopman_generator

    def _compute_lifted_outputs(self, z: NDArray) -> NDArray:
        """Gets the outputs for the Koopman matrix estimator. In this case, the
        outputs are the state measurements.

        Arguments:
            z: state vector

        Returns:
            outputs: vector of outputs to be lifted in basis

        """
        outputs = z + self.xdot_meas * self._dt

        return self.compute_basis_functions(outputs)


class KoopmanGeneratorEstimator(KoopmanEstimator):
    """Interface to the parameter adaptation law estimating the Koopman generator.

    Methods:
        TBD

    """

    def __init__(
        self,
        n_states: int,
        dt: float,
        nominal_f: callable,
        nominal_g: callable,
        use_rnn: bool = False,
    ):
        """Class initializer.

        Arguments:
            n_states: number of states (i.e. dimension of vector field)
            n_params: number of basis functions

        """
        super().__init__(n_states, dt, nominal_f, nominal_g, use_rnn)

    def _get_koopman_generator(self) -> NDArray:
        """Obtains the Koopman matrix from the adapted parameters.

        Arguments:
            None

        Returns:
            koopman_matrix

        """
        self.koopman_generator = self.theta.reshape((self.n_params, self.n_params)).T

        return self.koopman_generator

    def _compute_lifted_outputs(self, z: NDArray) -> NDArray:
        """Gets the outputs for the Koopman matrix estimator. In this case, the
        outputs are the state derivative measurements.

        Arguments:
            z: state vector

        Returns:
            outputs: vector of outputs to be lifted in basis

        """
        dpxdx = self.compute_basis_function_gradients(z)

        return dpxdx @ self.xdot_meas


class DataDrivenKoopmanMatrixEstimator(KoopmanEstimator):
    """Class interface to the data driven Koopman matrix estimation algorithm.

    Methods:
        estimate_uncertainty_lstsq

    """

    def __init__(self, n_states: int, dt: float):
        """Class initializer.

        Arguments:
            n_states: number of states
            dt: timestep size

        """
        super().__init__(n_states, dt)

        # Deques for testing least squares approach
        self.PX = deque([], maxlen=500)
        self.PY = deque([], maxlen=500)
        self.DPXDX = deque([], maxlen=500)

    #! Not usable right now, need to update for new Class
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

        # Deque to numpy array
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
