"""Test conjugate gradient implementations"""

import numpy as np
from scipy.sparse import linalg

from IMPRSPython.optimiser_func_set import derivative_rosenbrock, rosenbrock

from IMPRSPython import cg, cg_class


def test_conjugate_gradient():
    # SPD matrix (5x5 Tridiagonal)
    A = np.array(
        [
            [4, 1, 0, 0, 0],
            [1, 4, 1, 0, 0],
            [0, 1, 4, 1, 0],
            [0, 0, 1, 4, 1],
            [0, 0, 0, 1, 4],
        ]
    )

    # Define b
    b = np.array([1, 2, 3, 4, 5])

    # Initial guess
    x0 = np.zeros_like(b)

    x_ref, info = linalg.cg(A, b, x0=x0)
    x, _ = cg.conjugate_gradient(A, x0, b)
    assert np.allclose(x, x_ref)


def test_nonlinear_conjugate_gradient():
    # Expected minimum for Rosenbrock
    x_min = np.array([1.0, 1.0])

    # Initial guess
    # For x0 = np.array([-1.0, 0.8]), this was not converging
    x0 = np.array([0.2, 0.5])

    x, n_iter = cg.nonlinear_conjugate_gradient(
        rosenbrock, derivative_rosenbrock, x0, max_iter=2000, tol=1.0e-5
    )
    assert n_iter == 1269, "Converged before hit max iterations"
    assert np.allclose(x, x_min, atol=1.0e-4)


def test_nonlinear_conjugate_gradient_obj():
    # Expected minimum for Rosenbrock
    x_min = np.array([1.0, 1.0])

    # Initial guess
    # For x0 = np.array([-1.0, 0.8]), this was not converging
    x0 = np.array([0.2, 0.5])

    nlcg = cg_class.NLCG(
        rosenbrock, derivative_rosenbrock, x0, max_iter=2000, tol=1.0e-5
    )
    x, n_iter = nlcg.minimize()
    assert n_iter == 1269, "Converged before hit max iterations"
    assert np.allclose(x, x_min, atol=1.0e-4)


def test_bfgs_optimiser():
    # Expected minimum for Rosenbrock
    x_min = np.array([1.0, 1.0])

    # Initial guess
    x0 = np.array([0.2, 0.5])

    x, n_iter = cg.bfgs_optimiser(
        rosenbrock, derivative_rosenbrock, x0, max_iter=3000, tol=1.0e-4
    )
    print(f"Func. {n_iter} iterations to get x_min", x)
    assert n_iter == 3000, "BFGS does not quite reach convergence"
    assert np.allclose(x, x_min, atol=1.0e-3)
    assert np.allclose(x, np.array([0.9999086, 0.99981565]))


def test_bfgs_obj():
    # Result from `test_bfgs_optimiser`
    x_ref = np.array([0.9999086, 0.99981565])

    # Initial guess
    x0 = np.array([0.2, 0.5])

    bfgs = cg_class.BFGS(
        rosenbrock, derivative_rosenbrock, x0, max_iter=3000, tol=1.0e-4
    )

    x, n_iter = bfgs.minimize()
    assert np.allclose(
        x, x_ref
    ), "Result should be the same as the function design"
