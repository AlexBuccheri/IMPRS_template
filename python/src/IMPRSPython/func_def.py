""" Function definitions for use with type-hinting
"""
from collections.abc import Callable
from typing import Tuple

import numpy as np


# Define a function that evaluates a vector to a float.
FuncType = Callable[[np.ndarray], float]

# Define a derivative that accepts a vector and returns a vector of the same length.
DerFuncType = Callable[[np.ndarray], np.ndarray]

# Result from a CG minimisation
CGResultType = Tuple[np.ndarray, int]


def is_positive_definite(a: np.ndarray) -> bool:
    """ Determine if a matrix is positive-definite.

    cholesky decomposition will fail if a matrix is not positive-definite.
    :param a: 2D array.
    :return: True or false.
    """
    try:
        np.linalg.cholesky(a)
        return True
    except np.linalg.LinAlgError():
        return False
