"""Functions to test unconstrained optimisation.
See https://en.wikipedia.org/wiki/Test_functions_for_optimization for more.
"""
import numpy as np


def rosenbrock(x: np.ndarray) -> float:
    """
    Rosenbrock function.

    Formula:
      f(x) = sum_{i=1}^{n-1} [ 100 (x_{i+1} - x_i^2)^2 + (x_i - 1)^2 ]

    Global minimum: f(x) = 0 at x = (1, 1, ..., 1)
    """
    x = np.asarray(x)
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)


def derivative_rosenbrock(x: np.ndarray) -> np.ndarray:
    """Derivative of the Rosenbrock function.

    :param x: Function argument vector
    :return: Derivative of the function
    """
    df = np.empty_like(x)
    df[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
    df[-1] = 200 * (x[-1] - x[-2] ** 2)

    if x.size > 2:
        df[1:-1] = (
            -400 * x[1:-1] * (x[2:] - x[1:-1] ** 2)
            - 2 * (1 - x[1:-1])
            + 200 * (x[1:-1] - x[:-2] ** 2)
        )
    return df
