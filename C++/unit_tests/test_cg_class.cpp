#define CATCH_CONFIG_MAIN

#include <catch2/catch_all.hpp>

#include "test_functions/test_functions.h"

#include "optimisers/cg_class.h"

TEST_CASE("NLCG Child Class", "[optimiser]") {
    using namespace optimiser::armadillo;

    // Initial guess
    const arma::vec x0 = {0.2, 0.5};

    // Python settings
    constexpr int max_iter = 2000;
    constexpr double tol=1.e-5;
    constexpr double reduction_factor = 0.5;
    constexpr double c1 = 1.e-4;

    // Result from the python implementation
    const arma::vec x_ref = {1.0, 1.0};

    // TODO Call NLCG class and minimise F
    const arma::vec x = arma::zeros(x0.size());
    constexpr int n_iter = 1;

    REQUIRE(arma::approx_equal(x, x_ref, "both", 1.e-6, 1.e-5));
    REQUIRE(n_iter == 1269);
}


TEST_CASE("BFGS Child Class", "[optimiser]") {
    using namespace optimiser::armadillo;

    // Initial guess
    const arma::vec x0 = {0.2, 0.5};

    // BFGS settings
    constexpr int max_iter = 3000;
    constexpr double tol=1.e-4;
    constexpr double reduction_factor = 0.5;
    constexpr double c1 = 1.e-4;

    // Result from the python implementation
    const arma::vec x_ref = {0.9999086, 0.99981565};

    // TODO Call BFGS class and minimise F
    const arma::vec x = arma::zeros(x0.size());
    constexpr int n_iter = 1;

    REQUIRE(arma::approx_equal(x, x_ref, "both", 1.e-6, 1.e-5));
    REQUIRE(n_iter == max_iter);
}



