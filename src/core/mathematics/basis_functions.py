"""basis_functions.py

This module contains basis functions and their relevant gradients.

Methods:
    basis_functions
    basis_function_grads

"""

from nptyping import NDArray
import symengine as se
import numpy as np

#! FIND A BETTER WAY TO DO THIS!!!
# Define Global variables
global _MONOMIALS, _SINUSOIDS, _RBFS, _XS
_MONOMIALS = None
_SINUSOIDS = None
_RBFS = None
_XS = None


def basis_functions(x: NDArray) -> NDArray:
    """Computes the values of the basis functiions evaluated at the current
    state. Returns the (b x 1) vector of basis function values.

    Arguments:
        x: input vector at which functions are evaluated

    Returns:
        basis_values: values of the basis functions evaluated at x

    """
    global _MONOMIALS, _SINUSOIDS, _RBFS, _XS
    # Generate symbolic wrappers
    if (_MONOMIALS is None) or (_SINUSOIDS is None) or (_RBFS is None) or (_XS is None):
        _MONOMIALS, _XS = _monomial_basis_functions(len(x))
        _SINUSOIDS, _ = _sinusoidal_basis_functions(len(x))
        _RBFS, _ = _radial_basis_functions(len(x))

    monomials = symbolic_bases_wrapper(_MONOMIALS, _XS)
    sinusoids = symbolic_bases_wrapper(_SINUSOIDS, _XS)
    rbfs = symbolic_bases_wrapper(_RBFS, _XS)

    # Compute function values and stack them
    basis_values = np.hstack([monomials(x), sinusoids(x)])

    # basis_values = monomials(x)
    basis_values = sinusoids(x)
    # basis_values = rbfs(x)

    return basis_values


def basis_function_gradients(x: NDArray) -> NDArray:
    """Computes the gradients of the basis functions at the input value x.

    Arguments:
        x: input vector

    Returns:
        gradients: gradients of the basis functions at x

    """
    global _MONOMIALS, _SINUSOIDS, _RBFS, _XS
    # Generate symbolic wrappers
    if (_MONOMIALS is None) or (_SINUSOIDS is None) or (_RBFS is None) or (_XS is None):
        _MONOMIALS, _XS = _monomial_basis_functions(len(x))
        _SINUSOIDS, _ = _sinusoidal_basis_functions(len(x))
        _RBFS, _ = _radial_basis_functions(len(x))

    # Generate gradients
    monomial_gradients = _basis_function_gradients(x, _XS, _MONOMIALS)
    sinusoidal_gradients = _basis_function_gradients(x, _XS, _SINUSOIDS)
    rbf_gradients = _basis_function_gradients(x, _XS, _RBFS)

    gradients = np.vstack([monomial_gradients, sinusoidal_gradients])

    # gradients = monomial_gradients
    gradients = sinusoidal_gradients
    # gradients = rbf_gradients

    return gradients


# "Private" Functions -- do not need to be accessed outside module
def _monomial_basis_functions(n_states: int):
    """Defines the monomial basis functions for the Koopman lifting procedure."""
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

    # Sin basis functions
    sin_1n = [2 ** (1 / 2) * se.sin(1 * np.pi * xs[ii]) for ii in range(len(xs))]
    sin_2n = [2 ** (1 / 2) * se.sin(2 * np.pi * xs[ii]) for ii in range(len(xs))]
    sin_3n = [2 ** (1 / 2) * se.sin(3 * np.pi * xs[ii]) for ii in range(len(xs))]
    sin_4n = [2 ** (1 / 2) * se.sin(4 * np.pi * xs[ii]) for ii in range(len(xs))]
    sin_5n = [2 ** (1 / 2) * se.sin(5 * np.pi * xs[ii]) for ii in range(len(xs))]
    sin_6n = [2 ** (1 / 2) * se.sin(6 * np.pi * xs[ii]) for ii in range(len(xs))]
    sin_7n = [2 ** (1 / 2) * se.sin(7 * np.pi * xs[ii]) for ii in range(len(xs))]
    sin_8n = [2 ** (1 / 2) * se.sin(8 * np.pi * xs[ii]) for ii in range(len(xs))]
    # sin_9n = [2 ** (1 / 2) * se.sin(9 * np.pi * xs[ii]) for ii in range(len(xs))]
    # sin_10n = [2 ** (1 / 2) * se.sin(10 * np.pi * xs[ii]) for ii in range(len(xs))]

    # Cos basis functions
    cos_1n = [2 ** (1 / 2) * se.cos(1 * np.pi * xs[ii]) for ii in range(len(xs))]
    cos_2n = [2 ** (1 / 2) * se.cos(2 * np.pi * xs[ii]) for ii in range(len(xs))]
    cos_3n = [2 ** (1 / 2) * se.cos(3 * np.pi * xs[ii]) for ii in range(len(xs))]
    cos_4n = [2 ** (1 / 2) * se.cos(4 * np.pi * xs[ii]) for ii in range(len(xs))]
    cos_5n = [2 ** (1 / 2) * se.cos(5 * np.pi * xs[ii]) for ii in range(len(xs))]
    cos_6n = [2 ** (1 / 2) * se.cos(6 * np.pi * xs[ii]) for ii in range(len(xs))]
    cos_7n = [2 ** (1 / 2) * se.cos(7 * np.pi * xs[ii]) for ii in range(len(xs))]
    cos_8n = [2 ** (1 / 2) * se.cos(8 * np.pi * xs[ii]) for ii in range(len(xs))]
    # cos_9n = [2 ** (1 / 2) * se.cos(9 * np.pi * xs[ii]) for ii in range(len(xs))]
    # cos_10n = [2 ** (1 / 2) * se.cos(10 * np.pi * xs[ii]) for ii in range(len(xs))]

    # sinusoids = (
    #     sin_1n + sin_2n + sin_3n + sin_4n + sin_5n + sin_6n + sin_7n + sin_8n + sin_9n + sin_10n
    # )

    # Testing
    sinusoids = (
        sin_1n
        + cos_1n
        + sin_2n
        + cos_2n
        + sin_3n
        + cos_3n
        + sin_4n
        + cos_4n
        + sin_5n
        + cos_5n
        + sin_6n
        + cos_6n
        + sin_7n
        + cos_7n
        + sin_8n
        + cos_8n
    )

    return se.Matrix(sinusoids), xs


def _radial_basis_functions(n_states: int):
    """Defines the radial basis functions for the Koopman lifting procedure."""
    # Define symbolic state
    xs = se.symbols([f"x{num+1}" for num in range(n_states)])

    # Define radial basis functions
    gain = 0.001
    # rbfs = [
    #     se.exp(-gain * (xs[ii] * xs[jj]) ** 2) for ii in range(len(xs)) for jj in range(len(xs))
    # ]
    gain = 0.1
    rbfs = [10 * se.exp(-gain * (xs[ii]) ** 2) for ii in range(len(xs))]

    return se.Matrix(rbfs), xs


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
