import numpy as np


def ramp(x: float, k: float, d: float) -> float:
    """Approximation to the unit ramp function.

    INPUTS
    ------
    x: independent variable
    k: scale factor
    d: offset factor

    OUTPUTS
    -------
    y = 1/2 + 1/2 * tanh(k * (x - d))

    """
    return 0.5 * (1 + np.tanh(k * (x - d)))


def dramp(x: float, k: float, d: float) -> float:
    """Derivative of approximation to the unit ramp function.

    INPUTS
    ------
    x: independent variable
    k: scale factor
    d: offset factor

    OUTPUTS
    -------
    dy = k/2 * (sech(k * (x - d))) ** 2

    """
    return k / (2 * (np.cosh(k * (x - d))) ** 2)


def d2ramp(x: float, k: float, d: float) -> float:
    """2nd Derivative of approximation to the unit ramp function.

    INPUTS
    ------
    x: independent variable
    k: scale factor
    d: offset factor

    OUTPUTS
    -------
    dy = k/2 * (sech(k * (x - d))) ** 2

    """
    arg = k * (d - x)
    return k**2 * np.tanh(arg) / np.cosh(arg) ** 2


def wrapToPi(x):
    xwrap = np.remainder(x, 2 * np.pi)
    mask = np.abs(xwrap) > np.pi
    xwrap[mask] -= 2 * np.pi * np.sign(xwrap[mask])
    mask1 = x < 0
    mask2 = np.remainder(x, np.pi) == 0
    mask3 = np.remainder(x, 2 * np.pi) != 0
    xwrap[mask1 & mask2 & mask3] -= 2 * np.pi
    return xwrap
