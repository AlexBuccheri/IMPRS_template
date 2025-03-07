#ifndef IMPRS_CG_BASE_CLASS_H
#define IMPRS_CG_BASE_CLASS_H
/*
 * Parent/base class for the non-linear CG algorithm.
 * Specific implementations, such as BFGS, can inherit from this
 * and implement their specialised methods without needing to reimplement
 * the CG algorithm, given in minimize().
 *
 * The class is not templated, so any non-virtual functions are defined
 * in the respective .cpp file to reduce compilation overhead.
 */
#include <functional>

#include <armadillo>

#include "cg.h"

namespace optimiser::armadillo::cg {

    class NonLinearCGBase {
    // Child classes can access protected members
    protected:
        // TODO. Add arguments
        // TODO. Add data that defines the internal state state

    public:
        /**
         * @brief Constructs a non-linear conjugate gradient optimiser.
         *
         * @param f The objective function to minimize.
         * @param df The derivative (gradient) of the objective function.
         * @param x0 The initial guess for the solution.
         * @param max_iter The maximum number of iterations allowed.
         * @param tol The tolerance for convergence.
         */
        NonLinearCGBase(const FuncType &f,
                    const DerFuncType &df,
                    const arma::vec &x0,
                    int max_iter,
                    double tol){
            // TODO Add a member initialisation list
        }

        /// Destructor
        ~NonLinearCGBase() = default;

        /// Generic implementation of non-linear optimiser.
        CGResult minimize();

        // Pure virtual functions (no base class implementations)
        /// Initializes the search direction.
        virtual arma::vec initialise_search_direction() = 0;

        /// Performs a line search.
        virtual double line_search() = 0;

        /// Updates the Hessian approximation or coefficient.
        virtual arma::mat update_hessian_or_coefficient() = 0;

        /// Updates the search direction.
        virtual arma::vec update_search_direction() = 0;
        };

}
#endif
