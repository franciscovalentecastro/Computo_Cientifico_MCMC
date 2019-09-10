#!/usr/bin/env python
# -*- coding: utf-8 -*-
from scipy import linalg
import numpy as np
from substitution import *
from factorization import *

import matplotlib.pyplot as plt


def polynomial_base(x, degree, coefficient):
    value = 0.0
    for idx in range(0, degree):
        value += coefficient[idx] * x**idx
    return value


def least_squares_estimator(X, Y):
    # Get QR decomposition of data matrix
    (Q, R) = qr_factorization(X)

    # Transform y vector
    Y_prime = Q.T @ Y.T

    # Solve system R * beta = y_prime
    beta = backward_substitution(R, Y_prime.T)

    return beta


def least_squares_estimator_scipy(X, Y):
    # Get QR decomposition of data matrix
    (Q, R) = linalg.qr(X, mode="economic")

    # Transform y vector
    Y_prime = Q.T @ Y.T

    # Solve system R * beta = y_prime
    beta = backward_substitution(R, Y_prime.T)

    return beta


def least_squares_polynomial_fit(X, Y, degree):
    # Get dimensions
    n = len(X)

    # Create Vandermonde matrix
    X_v = np.ones((n, degree), dtype=np.float)
    for idx in range(1, degree):
        X_v[:, idx] = X * X_v[:, idx - 1]

    # Obtain least square estimator
    beta = least_squares_estimator(X_v, Y)

    # Create polynomial fucntion
    def polynomial(x):
        return polynomial_base(x, degree, beta)

    return polynomial


def least_squares_polynomial_fit_scipy(X, Y, degree):
    # Get dimensions
    n = len(X)

    # Create Vandermonde matrix
    X_v = np.ones((n, degree), dtype=np.float)
    for idx in range(1, degree):
        X_v[:, idx] = X * X_v[:, idx - 1]

    # Obtain least square estimator
    beta = least_squares_estimator_scipy(X_v, Y)

    # Create polynomial fucntion
    def polynomial(x):
        return polynomial_base(x, degree, beta)

    return polynomial


def generate_sin_curve_data(size, sigma):
    # Create data vectors
    X = np.zeros(size, dtype=np.float)
    Y = np.random.normal(loc=0.0, scale=sigma, size=size)

    # Generate random data from curve sen(4*pi*i)
    for idx in range(0, size):
        X[idx] = (4.0 * np.pi * float(idx)) / float(size)
        Y[idx] += np.sin(X[idx])

    return (X.astype(np.float), Y.astype(np.float))


def main():
    # Set parameters
    n = 100
    sigma = 0.11
    degree = 10
    epsilon = .1

    # Generate data on sin curve
    (X, Y) = generate_sin_curve_data(n, sigma)

    # Get polynomial fit with the least square estimator
    polynomial = least_squares_polynomial_fit(X, Y, degree)

    # Plot fit
    x_range = np.arange(epsilon, 4 * np.pi - epsilon, epsilon)
    y_range = np.zeros(len(x_range), dtype=np.float)

    for idx in range(0, len(x_range)):
        y_range[idx] = polynomial(x_range[idx])

    plt.plot(X, Y, ".")
    plt.plot(x_range, y_range)
    plt.show()


if __name__ == "__main__":
    main()
