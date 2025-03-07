#include <cassert>

#include <armadillo>

#include "integration.h"

namespace integration::armadillo {

    double trapezium(const FuncType1D &f, const arma::vec &x, const double dx){
        // Hints:
        // 1. Evaluate f at x
        // 2. use arma::sum and .subvec() as an efficient means of summing over the required elements
        return 0.0;
    }

    double trapezium(const FuncType1D &f, const Limits &limits, const double dx){
        // Hints:
        // use regspace to define x
        // return a call to the function above
        return 0.0;
    }

    double simpson(const FuncType1D &f, const Limits &limits, const int npoints){
        assert(npoints % 2 == 1);
        const arma::vec x = arma::linspace(limits.start, limits.end, npoints);
        return simpson(f, x, npoints);
    }

    double simpson(const FuncType1D &f, const arma::vec &x, const int npoints){
        // Hints:
        // Assert npoints is odd, as we require the number of subintervals to be even
        // Note the definition n_subintervals = npoints - 1;
        // Use arma::linspace<arma::uvec> to create index maps for the two sets of summations
        // For example, f_x.elem(odd_indices) should return all elements in the sum:
        // \sum_{i=1}^{n / 2} f\left(x_{2 i-1}\right) = f(x1) + f(x3) + ... f(x_{n-1})
        return 0.0;
    }
}
