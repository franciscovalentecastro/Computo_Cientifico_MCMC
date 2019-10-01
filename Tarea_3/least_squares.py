#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import numpy as np
from scipy import linalg

from substitution import *
from factorization import *


def least_squares_estimator(X, Y):
    # Get QR decomposition of data matrix
    (Q, R) = qr_factorization(X)

    print(Q)
    print(R)

    print("QR factorization correct ? : ",
          np.allclose(Q @ R, X), end="\n\n")

    # Transform y vector
    Y_prime = Q.T @ Y

    # Solve system R * beta = y_prime
    beta = backward_substitution(R, Y_prime.T)

    return beta


def least_squares_estimator_scipy_qr(X, Y):
    # Get QR decomposition of data matrix
    (Q, R) = linalg.qr(X, mode="economic")

    # Transform y vector
    Y_prime = Q.T @ Y

    # Solve system R * beta = y_prime
    beta = backward_substitution(R, Y_prime.T)

    return beta


def least_squares_estimator_scipy_inv(X, Y):
    # Obtain inverse of matrix
    Xt_X_inv = linalg.inv(X.T @ X)

    print(Xt_X_inv.shape)
    print(X.T.shape)
    print(Y.shape)

    # Solve system R * beta = y_prime
    beta = Xt_X_inv @ X.T @ Y

    return beta


def generate_linear_data(X, beta, size, sigma):
    # Create data vectors
    Y = np.random.normal(loc=0.0, scale=sigma, size=size)

    # Add linear data
    print(X.shape)
    print(beta.shape)
    Y += X @ beta

    return Y


def main():
    # Set parameters
    n = 20
    sigma = 0.15
    sigma_prime = .01
    degree = 5
    beta = np.array([5, 4, 3, 2, 1]).T
    dimension = (n, degree)

    colinear = False
    if(len(sys.argv) > 1):
        colinear = True if sys.argv[1] == "True" else False
    else:
        print("Not enough parameters.")

    # Generate random matrix X
    if not colinear:
        X = generate_random_matrix(dimension)
    else:
        # Generate colinear data
        X = generate_random_matrix(dimension)
        for idx in range(1, 5):
            X[:, idx] = (idx + 1) * X[:, 0] \
                + np.random.normal(loc=0.0, scale=.01, size=n)

    print(X)

    # Add noise delta to matrix
    X_delta = X + np.random.normal(loc=0.0, scale=sigma_prime, size=dimension)

    # Generate linear data
    Y = generate_linear_data(X, beta, n, sigma)

    # Get least squares estimator
    b1 = least_squares_estimator(X, Y)
    b2 = least_squares_estimator(X_delta, Y)
    b3 = least_squares_estimator_scipy_inv(X_delta, Y)
    b4 = least_squares_estimator_scipy_qr(X_delta, Y)

    print(beta)
    print(b1)
    print(b2)
    print(b3)
    print(b4)

    print(np.linalg.cond(X))
    print(np.linalg.cond(X_delta))

    # Get polynomial fit with the least square estimator
    # polynomial = least_squares_polynomial_fit(X, Y, degree)

    # # Plot fit
    # x_range = np.arange(epsilon, 4 * np.pi - epsilon, epsilon)
    # y_range = np.zeros(len(x_range), dtype=np.float)

    # for idx in range(0, len(x_range)):
    #     y_range[idx] = polynomial(x_range[idx])

    # plt.plot(X, Y, ".")
    # plt.plot(x_range, y_range)
    # plt.show()


if __name__ == "__main__":
    main()
