from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import List, Optional, Tuple

import numpy as np

from IMPRSPython.func_def import FuncType, DerFuncType, CGResultType


class NLConjugateGradient(ABC):

    def __init__(
        self,
        f: FuncType,
        df: DerFuncType,
        x0: np.ndarray,
        max_iter: Optional[int] = 500,
        tol: Optional[float] = 1.0e-6,
        hooks: Optional[List[Callable]] = None,
    ):
        """Initialise variables prior to CG loop"""
        self.f = f
        self.df = df
        self.max_iter = max_iter
        self.tol = tol
        if hooks is None:
            self.hooks = []

        # Initialise variable
        self.x = np.copy(x0)
        # Compute gradient
        self.g = self.df(self.x)
        # Initialise step size
        self.alpha = 1
        # Zero the search direction
        self.d = 0
        # Zero coefficient or approx inv Hessian
        self.hess = 0
        # Zero iteration counter
        self.k = 0

    @abstractmethod
    def initialise_search_direction(self) -> float:
        """Initialise the search direction,

        Uses the current gradient, self.g, and (if present) the inverse Hessian approximation
        self.hess, to form the descent direction.

        self.hess step preconditions the gradient by the approximate curvature information,
        leading to a more informed search direction compared to using just the negative gradient.
        """
        pass

    @abstractmethod
    def line_search(self) -> float:
        """Perform a line search to determine a suitable step length, self.alpha,
        along the current search direction.
        """
        pass

    @abstractmethod
    def update_hessian_or_coefficient(self) -> float | np.ndarray:
        """Depending on the choice of algorithm up the coefficient or approximate
        inverse Hessian (both represented by self.hess).
        """
        pass

    @abstractmethod
    def update_search_direction(self) -> np.ndarray:
        """Update the search direction, self.d
        This should be mathematically equivalent to initialise_search_direction,
        but use self.g_next, as the vectors x and gradient g updated AFTER the search direction.
        """
        pass

    def minimize(self) -> CGResultType:
        """Minimize f(x) using non-linear CG"""
        self.d = self.initialise_search_direction()

        for k in range(0, self.max_iter):
            self.k = k

            # Compute new step length
            self.alpha = self.line_search()

            # Compute new variable
            self.x_next = self.x + self.alpha * self.d

            # Compute new gradient
            self.g_next = self.df(self.x_next)

            # Call any optional functions prior to updating the Hessian/coefficient
            for func in self.hooks:
                func()

            # Update coefficient or approx inv Hessian
            self.hess = self.update_hessian_or_coefficient()

            if np.linalg.norm(self.g_next) <= self.tol:
                return self.x_next, self.k

            # Update quantities
            self.d = self.update_search_direction()
            self.x = self.x_next
            self.g = self.g_next

        return self.x, self.max_iter
