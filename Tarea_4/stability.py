#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy import linalg

import sys
import timeit
import functools

import matplotlib.pyplot as plt

from substitution import *
from factorization import *
from least_squares import *


def measure_factorization_time(matrix, method, number=1, **kwargs):
    # Calculate sum of execution times of "number".
    # Return the mean of the execution time when dividing by "number".
    return timeit.timeit(functools.partial(method, matrix),
                         number=number, **kwargs) / number


def plot_times(times_cholesky_b, times_cholesky_b_e,
               times_scipy_b, times_scipy_b_e, parameters):
    x_range = range(len(times_cholesky_b))

    # Plot times
    plt.plot(x_range, times_cholesky_b, ".-", label="Own Algorithm B")
    plt.plot(x_range, times_cholesky_b_e, ".-", label="Own Algorithm Be")
    plt.plot(x_range, times_scipy_b, ".-", label="Scipy Algorithm B")
    plt.plot(x_range, times_scipy_b_e, ".-", label="Scipy Algorithm Be")

    # Construct plot
    plt.legend(loc='upper right')
    # plt.xticks(x_range, [str(x) for x in parameters])
    # plt.yscale('log')
    plt.savefig('execution-times-N=%d' % len(parameters), bbox_inches='tight')
    plt.show()


def plot_errors(errors_a, errors_b, parameters):
    x_range = range(len(errors_a))

    # Plot error
    plt.plot(x_range, errors_a, ".-", label="Errors A", alpha=0.7)
    plt.plot(x_range, errors_b, ".-", label="Errors B", alpha=0.7)

    # Construct plot
    plt.legend(loc='upper right')
    # plt.xticks(x_range, [str(x) for x in parameters])
    plt.yscale('log')
    plt.savefig('error-N=%d' % len(parameters), bbox_inches='tight')
    plt.show()


def plot_condition(condition_b, condition_b_e, parameters):
    x_range = range(len(condition_b))

    # Plot error
    plt.plot(x_range, condition_b_e, ".-", label="Condition Be")
    plt.plot(x_range, condition_b, ".-", label="Condition B")

    # Construct plot
    plt.legend(loc='upper right')
    # plt.xticks(x_range, [str(x) for x in parameters])
    # plt.yscale('log')
    plt.savefig('condition-N=%d' % len(parameters), bbox_inches='tight')
    plt.show()


def compare_execution_speed(parameters):
    # Sample of times
    times_ls = []
    times_scipy = []

    # Dictionary of polynomial fit
    polynomial_fit_ls = {}
    polynomial_fit_scipy = {}

    # Define normal distribution
    sigma = 0.11

    # Loop trough different polynomial degrees and sample size
    for (degree, size) in parameters:

        # Generate data on sin curve
        (X, Y) = generate_sin_curve_data(size, sigma)

        # Measure least squares algorithm time using timeit
        tLS = measure_least_squares_time(X, Y, degree,
                                         least_squares_polynomial_fit)
        times_ls.append(tLS)

        # Get polynomial fit with the least square estimator
        polynomial_ls = least_squares_polynomial_fit(X, Y, degree)

        # Measure least squares using scipy algorithm
        tScipy = measure_least_squares_time(X, Y, degree,
                                            least_squares_polynomial_fit_scipy)
        times_scipy.append(tScipy)

        # Get polynomial fit with the least square estimator
        polynomial_scipy = least_squares_polynomial_fit_scipy(X, Y, degree)

        # Save polynomial aproximations
        polynomial_fit_ls[(degree, size)] = polynomial_ls
        polynomial_fit_scipy[(degree, size)] = polynomial_scipy

        # Plot both estimations
        plot_fit(X, Y, polynomial_ls, polynomial_scipy,
                 degree, size, tLS, tScipy)

    # Multiple degrees in same plot
    sizes = list(set([x[1] for x in polynomial_fit_ls.keys()]))

    for size in sizes:
        # Generate data on sin curve
        (X, Y) = generate_sin_curve_data(size, sigma)

        # Subset of dictionary with same size
        poly_dict = {x: polynomial_fit_ls[x]
                     for x in polynomial_fit_ls.keys()
                     if x[1] == size}

        # Plot multiple
        plot_multiple_fit(X, Y, poly_dict)

    # Plot execution times
    plot_times(times_ls, times_scipy, parameters)


def main():
    # Print format to 3 decimal spaces and fix seed
    # np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    largest_eigen_value = 50
    if(len(sys.argv) > 1):
        largest_eigen_value = float(sys.argv[1])
    else:
        print("Not enough parameters.")

    # Matrix parameters
    n = 20
    m = 50
    matrix_shape = (n, m)

    # Generate random matrix
    A = generate_random_matrix(matrix_shape)

    # Calculate QR factorization of A
    # print("Cond A", np.linalg.cond(A))
    (Q, _) = linalg.qr(A)

    # Set list of max eigenvalues
    max_eigen_values = np.arange(10, largest_eigen_value, 10)

    # Sample of times list
    times_Cholesky_B = []
    times_Cholesky_B_epsilon = []
    times_Scipy_B = []
    times_Scipy_B_epsilon = []

    errors_Cholesky_B = []
    errors_Cholesky_B_epsilon = []
    errors_Scipy_B = []
    errors_Scipy_B_epsilon = []

    condition_B = []
    condition_B_epsilon = []

    printStep = 100

    for eig_max in max_eigen_values:

        # Generate linear eigen values
        eigen_values = np.linspace(eig_max, 1.0, n)

        # Add gaussian noise to eigen values
        sigma = 0.11
        size = len(eigen_values)
        eigen_values_noisy = eigen_values + \
            np.random.normal(loc=0.0, scale=sigma, size=size)

        # Generate diferently condition matrices
        B = Q.T @ np.diag(eigen_values) @ Q
        B_epsilon = Q.T @ np.diag(eigen_values_noisy) @ Q

        condition_B.append(np.linalg.cond(B))
        condition_B_epsilon.append(np.linalg.cond(B_epsilon))

        tCholesky_B \
            = measure_factorization_time(B, cholesky_factorization)
        tCholesky_B_epsilon \
            = measure_factorization_time(B_epsilon, cholesky_factorization)
        tScipy_B \
            = measure_factorization_time(B, linalg.cholesky)
        tScipy_B_epsilon \
            = measure_factorization_time(B_epsilon, linalg.cholesky)

        times_Cholesky_B.append(tCholesky_B)
        times_Cholesky_B_epsilon.append(tCholesky_B_epsilon)
        times_Scipy_B.append(tScipy_B)
        times_Scipy_B_epsilon.append(tScipy_B_epsilon)

        # Measure error of factorization
        R = cholesky_factorization(B)
        errors_Cholesky_B.append(
            np.linalg.norm(B - R.T @ R))
        # print("Cholesky factorization correct ? : ",
        #        np.allclose(R.T @ R, B), end="\n\n")

        R_epsilon = cholesky_factorization(B_epsilon)
        errors_Cholesky_B_epsilon.append(
            np.linalg.norm(B - R_epsilon.T @ R_epsilon))

        R_scipy = linalg.cholesky(B)
        errors_Scipy_B.append(
            np.linalg.norm(B - R_scipy.T @ R_scipy))
        # print("Cholesky scipy factorization correct ? : ",
        #       np.allclose(R_scipy.T @ R_scipy, B), end="\n\n")

        R_epsilon_scipy = linalg.cholesky(B_epsilon)
        errors_Scipy_B_epsilon.append(
            np.linalg.norm(B - R_epsilon_scipy.T @ R_epsilon_scipy))

        # Every printStep iterations show current time information.
        if eig_max % printStep == 0:
            print("Cond : %.5f" % np.linalg.cond(B),
                  "Cholesky Time B : %.5f" % tCholesky_B,
                  "Scipy Time B : %.5f" % tScipy_B, sep=', ')

    # Plot errors
    plot_errors(errors_Cholesky_B,
                errors_Scipy_B,
                max_eigen_values)

    plot_errors(errors_Cholesky_B_epsilon,
                errors_Scipy_B_epsilon,
                max_eigen_values)

    plot_condition(condition_B,
                   condition_B_epsilon,
                   max_eigen_values)

    plot_errors(np.abs(np.array(errors_Cholesky_B) -
                       np.array(errors_Scipy_B)),
                np.abs(np.array(errors_Cholesky_B_epsilon) -
                       np.array(errors_Scipy_B_epsilon)),
                max_eigen_values)

    # Plot times
    plot_times(times_Cholesky_B,
               times_Cholesky_B_epsilon,
               times_Scipy_B,
               times_Scipy_B_epsilon,
               max_eigen_values)


if __name__ == "__main__":
    main()
