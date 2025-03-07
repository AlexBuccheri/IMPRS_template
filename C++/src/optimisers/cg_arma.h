#ifndef IMRESS_CG_H
#define IMRESS_CG_H

#include <cmath>
#include <functional>
#include <iostream>
#include <vector>

#include <armadillo>

#include "cg.h" // FuncType, DerFuncType, CGResult
#include "line_search.h" // LineSearchFunc

namespace optimiser::armadillo::cg {

    namespace linear{
        /**
         * @brief Solves the linear system of equations A * x = b using the
         *        Linear Conjugate Gradient method.
         *
         * This function iteratively solves the system of equations where `A` is a
         * positive semi-definite matrix, `b` is the known right-hand side vector,
         * and `x0` is an initial guess for the solution vector.
         *
         * @param a Positive semi-definite matrix (A).
         * @param x0 Initial guess for the left-hand side solution vector (x).
         * @param b Right-hand side solution vector (b).
         * @param tol Optional tolerance for convergence .
         * @param max_iter Optional maximum number of iterations.
         * @return Solution vector (x).
         */
        CGResult linear_conjugate_gradient(const arma::mat& a,
                                            const arma::vec& x0,
                                            const arma::vec& b,
                                            double tol = 1e-8,
                                            int max_iter = 1000);

    }

    namespace nonlinear{

        /**
         * @brief Performs non-linear conjugate gradient optimization.
         *
         * This function implements a non-linear conjugate gradient algorithm to minimize the given function.
         * It utilizes the provided objective function, its gradient, an initial guess, and a line search function
         * to determine appropriate step sizes. The algorithm iterates until either the maximum number of iterations
         * is reached or the specified tolerance is achieved.
         *
         * @param f The objective function to be minimized.
         * @param df The gradient (derivative) of the objective function.
         * @param x0 The initial guess vector.
         * @param line_search_func The line search function used to determine the step size.
         * @param max_iter The maximum number of iterations allowed (default is 1000).
         * @param tol The tolerance for convergence (default is 1.e-6).
         *
         * @return CGResult A structure containing the results of the optimization process, including the final solution,
         * convergence status, and any diagnostic information.
         */
        CGResult non_linear_conjugate_gradient(const FuncType& f,
                                               const DerFuncType& df,
                                               const arma::vec& x0,
                                               LineSearchFunc&,
                                               int max_iter = 1000,
                                               double tol = 1.e-6);
    }

    namespace bfgs {

        /**
         * @brief Update the inverse Hessian approximation in BFGS.
         *
         * This function updates the approximate inverse Hessian matrix using the BFGS update formula.
         * Given the differences in the iterates and gradients, the update is performed by
         *
         * \f[
         * H_{k+1} = \left(I - \rho_k s y^T\right) H_k \left(I - \rho_k y s^T\right) + \rho_k s s^T,
         * \quad \text{with} \quad \rho_k = \frac{1}{y^T s},
         * \f]
         *
         * ensuring the updated matrix maintains symmetry and, under appropriate conditions,
         * positive definiteness.
         *
         * @param s The difference between successive iterate vectors, i.e., \( s = x_{k+1} - x_k \).
         * @param y The difference between successive gradient vectors, i.e., \( y = \nabla f(x_{k+1}) - \nabla f(x_k) \).
         * @param hess The current approximate inverse Hessian matrix.
         * @return arma::mat The updated approximate inverse Hessian matrix.
         */
         arma::mat update_hessian(const arma::vec &s,
                                 const arma::vec &y,
                                 const arma::mat &hess);


        /**
         * @brief Broyden–Fletcher–Goldfarb–Shanno (BFGS) optimization.
         *
         * This function performs optimization using the BFGS quasi-Newton method. It iteratively updates
         * an approximate inverse Hessian to better inform the search direction and employs a line search
         * to compute an appropriate step size.
         *
         * @param f The objective function to be minimized.
         * @param df The derivative (gradient) function of the objective.
         * @param x0 The initial guess vector.
         * @param line_search_func The line search function used to compute the step length.
         * @param max_iter The maximum number of iterations allowed.
         * @param tol The tolerance for convergence.
         * @return CGResult A structure containing the results of the optimization process.
         */
        CGResult broyden_fletcher_goldfarb_shanno(const FuncType &f,
                                                  const DerFuncType &df,
                                                  const arma::vec &x0,
                                                  LineSearchFunc &line_search_func,
                                                  int max_iter = 1000,
                                                  double tol = 1.e-6);

    }

}

#endif
