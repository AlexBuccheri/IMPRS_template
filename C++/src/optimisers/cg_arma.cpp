#include <armadillo>

#include "line_search.h"

#include "cg_arma.h"

namespace optimiser::armadillo::cg {

    namespace linear {
        CGResult linear_conjugate_gradient(const arma::mat &a,
                                            const arma::vec &x0,
                                            const arma::vec &b,
                                            const double tol,
                                            const int max_iter) {
            // TODO Fill in the implementation
            return CGResult{x0, max_iter};
        }
    } // linear

    namespace nonlinear {
        double fletcher_reeves_coefficient(const arma::vec& grad, const arma::vec& grad_next)
        {
            return arma::dot(grad_next, grad_next) / arma::dot(grad, grad);
        }

        CGResult non_linear_conjugate_gradient(const FuncType &f,
                                               const DerFuncType &df,
                                               const arma::vec &x0,
                                               LineSearchFunc &line_search_func,
                                               const int max_iter,
                                               const double tol) {
            // TODO Fill in the implementation
            return {x0, max_iter};
        }
    } // nonlinear

    namespace bfgs {

        arma::mat update_hessian(const arma::vec &s,
                                 const arma::vec &y,
                                 const arma::mat &hess){
            // TODO Fill in the implementation
            return arma::mat{};
        }

        CGResult broyden_fletcher_goldfarb_shanno(const FuncType &f,
                                                  const DerFuncType &df,
                                                  const arma::vec &x0,
                                                  LineSearchFunc &line_search_func,
                                                  const int max_iter,
                                                  const double tol) {
            // TODO Fill in the implementation
            return {x0, max_iter};
        }

    } // bfgs

}
