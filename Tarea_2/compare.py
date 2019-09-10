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


def plot_times(times_ls, times_scipy, parameters):
    x_range = range(len(times_ls))

    # Plot times
    plt.plot(x_range, times_ls, ".-", label="Own Algorithm")
    plt.plot(x_range, times_scipy, ".-", label="Scipy Algorithm")

    # Construct plot
    plt.legend(loc='upper right')
    plt.xticks(x_range, [str(x) for x in parameters])
    plt.yscale('log')
    plt.savefig('execution-times-%d' % len(parameters), bbox_inches='tight')
    plt.show()


def plot_fit(X, Y, polynomial_ls, polynomial_scipy,
             degree, size, time_ls, time_scipy):
    # Plot fit from least squares
    epsilon = 0.05
    x_range = np.arange(epsilon, 12 - epsilon, epsilon)
    y_range = np.zeros(len(x_range), dtype=np.float)

    for idx in range(0, len(x_range)):
        y_range[idx] = polynomial_ls(x_range[idx])

    plt.subplot(1, 2, 1)
    plt.gca().set_title('Own Algorithm P=%d, N=%d \n' % (degree, size) +
                        'Execution Time=%f s' % time_ls, fontsize=10)
    plt.plot(X, Y, ".")
    plt.plot(x_range, y_range)

    # Plot polynomial fit from scipy module
    y_range_scipy = np.zeros(len(x_range), dtype=np.float)

    for idx in range(0, len(x_range)):
        y_range_scipy[idx] = polynomial_scipy(x_range[idx])

    plt.subplot(1, 2, 2)
    plt.gca().set_title('Scipy Algorithm P=%d, N=%d \n' % (degree, size) +
                        'Execution Time=%f s' % time_scipy, fontsize=10)
    plt.plot(X, Y, ".")
    plt.plot(x_range, y_range_scipy)
    plt.savefig('comparison-Deg=%d,N=%d' % (degree, size),
                bbox_inches='tight')
    plt.show()


def plot_multiple_fit(X, Y, polynomial_fit_dictionary):
    # Plot fit from least squares
    epsilon = 0.05
    x_range = np.arange(epsilon, 12 - epsilon, epsilon)

    # Plot data
    plt.plot(X, Y, ".")

    # Iterate through dictionary
    for key in polynomial_fit_dictionary.keys():
        (degree, size) = key

        # Evaluate polynomial
        y_range = np.zeros(len(x_range), dtype=np.float)
        for idx in range(0, len(x_range)):
            y_range[idx] = polynomial_fit_dictionary[key](x_range[idx])

        # Plot polynomial
        plt.plot(x_range, y_range, label="P=%d" % degree)

    # Construct plot
    plt.gca().set_title('Polynomial Fit N=%d \n' % size, fontsize=10)
    plt.legend(loc='upper right')
    plt.savefig('polynomial-fit-N=%d' % size, bbox_inches='tight')
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
        # Subset of dictionary with same size
        poly_dict = {x: polynomial_fit_ls[x]
                     for x in polynomial_fit_ls.keys()
                     if x[1] == size}

        # Plot multiple
        plot_multiple_fit(X, Y, poly_dict)

    # Plot execution times
    plot_times(times_ls, times_scipy, parameters)


def main():
    # Set up default parameters
    degrees = [3, 4, 6, 100]
    sizes = [100, 1000, 10000]

    # Create parameters list
    parameters = [(x, y) for y in sizes for x in degrees]

    compare_execution_speed(parameters)

    # Create parameters p = 0.1 * n
    parameters = []

    for n in [100, 1000, 2000, 2500, 2800, 2810]:
        p = int(0.1 * n)
        parameters.append((p, n))

    compare_execution_speed(parameters)


if __name__ == "__main__":
    main()
