#define CATCH_CONFIG_MAIN

#include <armadillo>
#include <catch2/catch_all.hpp>

#include "optimisers/cg_arma.h"
#include "test_functions/test_functions.h"

TEST_CASE("Linear Conjugate Gradient", "[optimiser]") {

    using namespace optimiser::armadillo::cg::linear;

    // SPD tri-diagonal matrix
    const arma::mat A = {
            {4, 1, 0, 0, 0},
            {1, 4, 1, 0, 0},
            {0, 1, 4, 1, 0},
            {0, 0, 1, 4, 1},
            {0, 0, 0, 1, 4}
    };
    const arma::vec b = {1, 2, 3, 4, 5};
    const arma::vec x0 = arma::zeros<arma::vec>(b.n_elem);
    const arma::vec x_ref = {0.16794872, 0.32820513, 0.51923077, 0.59487179, 1.10128205};

    const auto result = linear_conjugate_gradient(A, x0, b);

    REQUIRE(arma::approx_equal(result.x, x_ref, "reldiff", 1e-6));
}

TEST_CASE("Non-Linear Conjugate Gradient", "[optimiser]") {

    using namespace optimiser::armadillo;

    // TODO Set f and df
    //const auto f =
    //const auto df =
    const arma::vec x0 = {0.2, 0.5};
    const arma::vec x_ref = {1.0, 1.0};

    // Pre-set values for the optional parameters
    constexpr double reduction_factor = 0.5;
    constexpr double c1 = 1.e-4;

    // Create a lambda that fixes the extra parameters.
    // TODO Add parameters to capture to []
    LineSearchFunc line_search_func =
            []
                    (const FuncType & f,
                     const DerFuncType & df,
                     const arma::vec & x,
                     const arma::vec & direction,
                     double alpha0) -> double {
                // TODO Add function call
                return alpha0;  // Placeholder return value;
            };

    // TODO Add call to CG
    //const auto cg_result =

    const arma::vec x{};
    constexpr int n_iter = 1;
    REQUIRE(arma::approx_equal(x, x_ref, "both", 1.e-5, 1.e-4));
    REQUIRE(n_iter == 1269);
}
