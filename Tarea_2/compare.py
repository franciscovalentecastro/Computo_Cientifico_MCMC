#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import timeit
import functools

import matplotlib.pyplot as plt

from least_squares import *


def measure_least_squares_time(X, Y, degree, method, number=1, **kwargs):
    # Calculate sum of execution times of "number".
    # Return the mean of the execution time when dividing by "number".
    return timeit.timeit(functools.partial(method, X, Y, degree),
                         number=number, **kwargs) / number


def plot_fit(X, Y, polynomial_ls, polynomial_scipy, degree, size):
    # Plot fit from least squares
    x_range = np.arange(0, math.pi * 4.0, .1)
    y_range = np.zeros(len(x_range), dtype=np.float)

    for idx in range(0, len(x_range)):
        y_range[idx] = polynomial_ls(x_range[idx])

    plt.subplot(1, 2, 1)
    plt.gca().set_title('Own Algorithm deg=%d, N=%d' % (degree, size),
                        fontsize=10)
    plt.plot(X, Y, ".")
    plt.plot(x_range, y_range)

    # Plot polynomial from scipy module
    y_range_scipy = np.zeros(len(x_range), dtype=np.float)

    for idx in range(0, len(x_range)):
        y_range_scipy[idx] = polynomial_scipy(x_range[idx])

    plt.subplot(1, 2, 2)
    plt.gca().set_title('Scipy Algorithm deg=%d, N=%d' % (degree, size),
                        fontsize=10)
    plt.plot(X, Y, ".")
    plt.plot(x_range, y_range_scipy)

    plt.show()


def compare_execution_speed(parameters):

    # Sample of times
    times_ls = []
    times_scipy = []

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

        # Print times
        print(tLS, tScipy)

        # Plot both estimations
        plot_fit(X, Y, polynomial_ls, polynomial_scipy, degree, size)


def main():
    # Set up default parameters
    degrees = [3, 4, 6, 100]
    sizes = [100, 1000, 10000]

    # Create parameters list
    parameters = [(x, y) for x in degrees for y in sizes]

    compare_execution_speed(parameters)

    # Create parameters p = 0.1 * n
    parameters = []

    for n in [10, 100, 1000, 10000, 100000]:
        p = int(0.1 * n)
        parameters.append((p, n))

    compare_execution_speed(parameters)


if __name__ == "__main__":
    main()
