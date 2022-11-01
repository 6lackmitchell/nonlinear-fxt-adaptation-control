"""basis_functions.py

This module contains basis functions and their relevant gradients.

Methods:
    basis_functions
    basis_function_grads

"""

from nptyping import NDArray
import symengine as se
import numpy as np

# Define Symbolic State


def _monomial_basis_functions(n_states: int):
    """Defines the basis functions for the Koopman lifting procedure."""
    # Define symbolic state
    xs = se.symbols([f"x{num+1}" for num in range(n_states)])

    # Define monomials
    order_1 = [xs[ii] for ii in range(len(xs))]
    order_2 = [xs[ii] ** 2 for ii in range(len(xs))]
    order_3 = [xs[ii] ** 3 for ii in range(len(xs))]
    order_0 = [1.0]
    cross_1 = [xs[ii] * xs[(ii + 1) % len(xs)] for ii in range(len(xs))]
    cross_2 = [xs[ii] ** 2 * xs[(ii + 1) % len(xs)] for ii in range(len(xs))]
    cross_3 = [np.product(xs)]

    monomials = order_1 + order_2 + order_3 + order_0 + cross_1 + cross_2 + cross_3

    return se.Matrix(monomials), xs


def _sinusoidal_basis_functions(n_states: int) -> NDArray:
    """Defines the sinusoidal basis functions for the Koopman lifting procedure.

    Arguments:
        n_states: number of states from which to construct bases

    Returns:
        sinusoids: array of sinusoidal basis functions

    """
    # Define symbolic state
    xs = se.symbols([f"x{num+1}" for num in range(n_states)])

    # Individual basis functions
    sin_0n = [se.sin(xs[ii]) for ii in range(len(xs))]
    cos_0n = [se.cos(xs[ii]) for ii in range(len(xs))]
    sin_1n = [se.sin(2 * np.pi * xs[ii]) for ii in range(len(xs))]
    cos_1n = [se.cos(2 * np.pi * xs[ii]) for ii in range(len(xs))]
    sin_2n = [se.sin(4 * np.pi * xs[ii]) for ii in range(len(xs))]
    cos_2n = [se.cos(4 * np.pi * xs[ii]) for ii in range(len(xs))]
    sin_3n = [se.sin(6 * np.pi * xs[ii]) for ii in range(len(xs))]
    cos_3n = [se.cos(6 * np.pi * xs[ii]) for ii in range(len(xs))]
    sin_4n = [se.sin(8 * np.pi * xs[ii]) for ii in range(len(xs))]
    cos_4n = [se.cos(8 * np.pi * xs[ii]) for ii in range(len(xs))]

    # Cross basis functions
    cross_1 = [
        se.sin(2 * np.pi * xs[ii]) * se.cos(2 * np.pi * xs[(ii + 1) % len(xs)])
        for ii in range(len(xs))
    ]
    cross_2 = [
        se.cos(2 * np.pi * xs[(ii - 1) % len(xs)]) * se.sin(2 * np.pi * xs[(ii + 1) % len(xs)])
        for ii in range(len(xs))
    ]
    cross_3 = [xs[(ii - 1) % len(xs)] * se.sin(2 * np.pi * xs[ii]) for ii in range(len(xs))]
    cross_4 = [xs[(ii + 1) % len(xs)] * se.cos(2 * np.pi * xs[ii]) for ii in range(len(xs))]

    sinusoids = (
        sin_0n
        + cos_0n
        + sin_1n
        + cos_1n
        + sin_2n
        + cos_2n
        + sin_3n
        + cos_3n
        + sin_4n
        + cos_4n
        + cross_1
        + cross_2
        + cross_3
        + cross_4
    )

    return se.Matrix(sinusoids), xs


def symbolic_bases_wrapper(symbolic_bases, symbolic_state):
    """Wrapper for symbolic basis functions

    Arguments:
        symbolic_bases: array of symbolic expressions to be evaluated

    Returns:
        bases: function for evaluating symbolic basis functions

    ."""

    def compute_basis_values(x: NDArray) -> NDArray:
        """Evaluates the symbolic basis functions provided by the symbolic_bases argument

        Arguments:
            x: the input variable to replace the symbolic variable in symbolic_bases

        Returns:
            basis_values: values of specified basis functions evaluated at x

        """
        basis_values = np.array(symbolic_bases.subs(dict(zip(symbolic_state, x))))

        return np.squeeze(basis_values.astype(np.float32))

    return compute_basis_values


def _basis_function_gradients(x: NDArray, xs: se.Matrix, bases: se.Matrix) -> NDArray:
    """Computes the gradients of the sinusoidal basis functions (sinusoids) at
    the specified input (x).

    Arguments:
        x: array of values to replace the symbols in sinusoids
        sinusoids: symbolic basis functions in matrix/array form

    Returns:
        gradients: values of the gradients of the basis functions at x

    """
    gradient_symbolic = bases.jacobian(se.DenseMatrix(xs))
    gradient = symbolic_bases_wrapper(gradient_symbolic, xs)

    return gradient(x)


def basis_functions(x: NDArray) -> NDArray:
    """Computes the values of the basis functiions evaluated at the current
    state. Returns the (b x 1) vector of basis function values.

    Arguments:
        x: input vector at which functions are evaluated

    Returns:
        basis_values: values of the basis functions evaluated at x

    """
    # Generate symbolic wrappers
    mono_wrapper, xs = _monomial_basis_functions(len(x))
    sinu_wrapper, _ = _sinusoidal_basis_functions(len(x))
    monomials = symbolic_bases_wrapper(mono_wrapper, xs)
    sinusoids = symbolic_bases_wrapper(sinu_wrapper, xs)

    # Compute function values and stack them
    basis_values = np.hstack([monomials(x), sinusoids(x)])

    return basis_values


def basis_function_gradients(x: NDArray) -> NDArray:
    """Computes the gradients of the basis functions at the input value x.

    Arguments:
        x: input vector

    Returns:
        gradients: gradients of the basis functions at x

    """
    # Generate symbolic wrappers
    mono_wrapper, xs = _monomial_basis_functions(len(x))
    sinu_wrapper, _ = _sinusoidal_basis_functions(len(x))

    # Generate gradients
    monomial_gradients = _basis_function_gradients(x, xs, mono_wrapper)
    sinusoidal_gradients = _basis_function_gradients(x, xs, sinu_wrapper)

    return np.vstack([monomial_gradients, sinusoidal_gradients])


# def basis_functions(z: NDArray, min_len: int) -> NDArray:
#     """Computes the values of the basis functions evaluated at the current
#     state and control values. Returns the (b x 1) vector of basis function
#     values. May offload this to another module.

#     Arguments
#         z: input vector (may be states and controls or outputs)
#         min_len: minimum length of input vector

#     Returns
#         basis_funcs: vector of values of basis functions

#     """

#     # Append zeros to input z if necessary
#     if len(z) < min_len:
#         z = np.concatenate([z, np.zeros((min_len - len(z),))])

#     normalization_factor = 1 if np.max(abs(z)) == 0 else np.max(abs(z))
#     z = z / normalization_factor

#     # Monomial basis functions
#     psi_1nn = z * normalization_factor  # 1st Order
#     psi_2nn = z**2  # 2nd order
#     psi_3nn = z**3  # 3rd Order
#     psi_4nn = z**3 + 2 * z**2 - 3 * z - 1  # Polynomial
#     psi_5nn = -(z**3) - 5 * z**2 + 4 * z + 2  # Polynomial
#     psi_6nn = 2 * z**3 - 2 * z**2 + 2 * z - 2  # Polynomial
#     psi_7nn = 1.0 * np.exp(-(z**2) / 1)
#     psi_8nn = 1.0 * np.exp(-(z**2) / 2)
#     psi_9nn = 1.0 * np.exp(-(z**2) / 3)

#     # # Radial Basis Functions (RBFs)
#     # k = 1.0  # Multiplier for RBF
#     # Q = 1 / k * np.eye(len(z))  # Exponential gain for RBF

#     # psi_6n1 = k * np.exp(-1 / 2 * (z @ Q @ z))  # Radial Basis Functions
#     # psi_7nn = -k * Q @ z * np.exp(-1 / 2 * (z @ Q @ z))  # Gradient of RBF wrt z

#     basis_funcs = np.hstack(
#         [
#             psi_1nn,
#             psi_2nn,
#             psi_3nn,
#             psi_4nn,
#             psi_5nn,
#             psi_6nn,
#             psi_7nn,
#             psi_8nn,
#             psi_9nn,
#         ]
#     )

#     # return psi_1nn
#     return basis_funcs


# #! Need to make this a symbolic function!
# def basis_function_grads(z: NDArray, min_len: int) -> NDArray:
#     """Computes the gradients of the basis functions evaluated at the current
#     state and control values. Returns the (b x n) matrix of basis function
#     gradients. May offload this to another module.

#     Arguments
#         z: input vector (may be states and controls or outputs)
#         min_len: minimum length of input vector

#     Returns
#         basis_funcs: matrix of gradients of basis functions

#     """
#     # Append zeros to input z if necessary
#     if len(z) < min_len:
#         z = np.concatenate([z, np.zeros((min_len - len(z),))])

#     normalization_factor = 1 if np.max(abs(z)) == 0 else np.max(abs(z))
#     z = z / normalization_factor

#     # Monomial basis functions
#     psi_1nn = np.diag(np.ones((len(z),)) * normalization_factor)  # 1st Order
#     psi_2nn = np.diag(2 * z)  # 2nd order
#     psi_3nn = np.diag(3 * z**2)  # 3rd Order

#     psi_4nn = np.diag(3 * z**2 + 4 * z - 3)  # Polynomial
#     psi_5nn = np.diag(-3 * (z**2) - 10 * z + 4)  # Polynomial
#     psi_6nn = np.diag(6 * z**2 - 4 * z + 2)  # Polynomial

#     psi_7nn = np.diag(-2.0 * z * np.exp(-(z**2) / 1))
#     psi_8nn = np.diag(-1.0 * z * np.exp(-(z**2) / 2))
#     psi_9nn = np.diag(-2.0 / 3.0 * z * np.exp(-(z**2) / 3))

#     basis_func_grads = np.vstack(
#         [
#             psi_1nn,
#             psi_2nn,
#             psi_3nn,
#             psi_4nn,
#             psi_5nn,
#             psi_6nn,
#             psi_7nn,
#             psi_8nn,
#             psi_9nn,
#         ]
#     )

#     # return psi_1nn
#     return basis_func_grads


# TESTING
if __name__ == "__main__":
    mono_wrapped = _monomial_basis_functions(3)
    mono = symbolic_bases_wrapper(mono_wrapped[0], mono_wrapped[1])
    sinu_wrapped = _sinusoidal_basis_functions(3)
    sinu = symbolic_bases_wrapper(sinu_wrapped[0], sinu_wrapped[1])

    x = np.array([1, 2, 3])
    print(basis_functions(x))
    print(basis_function_gradients(x))

    # mono_bases = mono(x)
    # sinu_bases = sinu(x)

    # basis_function_gradients(x, mono_wrapped[1], mono_wrapped[0])

    # sinu = _sinusoidal_basis_functions(3)
    # print(sinu)
