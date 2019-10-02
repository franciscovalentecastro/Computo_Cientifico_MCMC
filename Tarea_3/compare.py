#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy import linalg

import timeit
import argparse
import functools

import matplotlib.pyplot as plt

from factorization import *

# Parser arguments
parser = argparse.ArgumentParser(description='Compare Cholesky Factorization')
parser.add_argument('--max-eigenvalue', '--me',
                    type=float, default=10, metavar='N',
                    help='The maximum eigenvalue to pick (default: 10)')
parser.add_argument('--step', '--s',
                    type=int, default=10, metavar='N',
                    help='Size of step between iterations (default: 10)')
parser.add_argument('--print-step', '--ps',
                    type=int, default=100, metavar='N',
                    help='Number of steps between stats log (default: 100)')
parser.add_argument('--generate-eigenvalues', '--ge',
                    default='linear',
                    choices=['linear', 'uniform'],
                    help='How are eigenvalues generated (default: "linear")')
args = parser.parse_args()


def measure_factorization_time(matrix, method, number=3, **kwargs):
    # Calculate sum of execution times of "number".
    # Return the mean of the execution time when dividing by "number".
    return timeit.timeit(functools.partial(method, matrix),
                         number=number, **kwargs) / number


def plot_times(x_range, times, labels):
    for idx, time in enumerate(times):
        plt.plot(x_range, time, ".-", label=labels[idx], alpha=0.7)
        plt.axhline(y=np.mean(time), color='black', linestyle='--')

    # Construct plot
    plt.legend(loc='upper right')
    plt.savefig('execution-times-N=%d' % len(x_range), bbox_inches='tight')
    plt.show()


def plot_errors(x_range, errors, labels):
    for idx, error in enumerate(errors):
        plt.plot(x_range, error, ".-", label=labels[idx], alpha=0.7)

    # Construct plot
    plt.legend(loc='upper right')
    plt.savefig('error-N=%d' % len(x_range), bbox_inches='tight')
    plt.show()


def plot_condition(x_range, conditions, labels):
    for idx, condition in enumerate(conditions):
        plt.plot(x_range, condition, ".-", label=labels[idx], alpha=0.7)

    # Construct plot
    plt.legend(loc='upper right')
    plt.savefig('condition-N=%d' % len(x_range), bbox_inches='tight')
    plt.show()


def compare_choklesky(matrix, max_eigenvalues):
    # Rename matrix
    Q = matrix

    # Lists to store samples
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

    # Loop through possible of eigenvalues
    for idx, eig_max in enumerate(max_eigenvalues):

        if args.generate_eigenvalues == 'linear':
            # Generate linear eigenvalues
            eigen_values = np.linspace(eig_max, 1.0, args.n)

        elif args.generate_eigenvalues == 'uniform':
            # Generate uniform eigenvalues
            uniform_sample = np.random.uniform(1, eig_max, args.n)
            uniform_sample[0] = 1
            uniform_sample[args.n - 1] = eig_max
            eigen_values = sorted(uniform_sample)[::-1]

        # Add gaussian noise to eigen values
        size = len(eigen_values)
        eigen_values_noisy = eigen_values + \
            np.random.normal(loc=0.0, scale=args.sigma, size=size)

        # Generate diferently conditioned matrices
        B = Q.T @ np.diag(eigen_values) @ Q
        B_epsilon = Q.T @ np.diag(eigen_values_noisy) @ Q

        # Calculate condition of B and Be
        condition_B.append(np.linalg.cond(B))
        condition_B_epsilon.append(np.linalg.cond(B_epsilon))

        # Measure the execution time of factorization
        times_Cholesky_B.append(
            measure_factorization_time(B, cholesky_factorization))
        times_Cholesky_B_epsilon.append(
            measure_factorization_time(B_epsilon, cholesky_factorization))
        times_Scipy_B.append(
            measure_factorization_time(B, linalg.cholesky))
        times_Scipy_B_epsilon.append(
            measure_factorization_time(B_epsilon, linalg.cholesky))

        # Cholesky of B error
        R = cholesky_factorization(B)
        errors_Cholesky_B.append(np.linalg.norm(B - R.T @ R))

        # Cholesky of Be error
        R_epsilon = cholesky_factorization(B_epsilon)
        errors_Cholesky_B_epsilon.append(
            np.linalg.norm(B_epsilon - R_epsilon.T @ R_epsilon))

        # Scipy of B error
        R_scipy = linalg.cholesky(B)
        errors_Scipy_B.append(np.linalg.norm(B - R_scipy.T @ R_scipy))

        # Scipy of Be error
        R_epsilon_scipy = linalg.cholesky(B_epsilon)
        errors_Scipy_B_epsilon.append(
            np.linalg.norm(B_epsilon - R_epsilon_scipy.T @ R_epsilon_scipy))

        # Every printStep iterations show current time information.
        if idx % args.print_step == 0:
            print("Cond : %.5f" % np.linalg.cond(B),
                  "Cholesky Time B : %.5f" % times_Cholesky_B[-1],
                  "Scipy Time B : %.5f" % times_Scipy_B[-1], sep=', ')

    # Return results
    times = [times_Cholesky_B, times_Cholesky_B_epsilon,
             times_Scipy_B, times_Scipy_B_epsilon]
    errors = [errors_Cholesky_B, errors_Cholesky_B_epsilon,
              errors_Scipy_B, errors_Scipy_B_epsilon]
    conditions = [condition_B, condition_B_epsilon]

    return (times, errors, conditions)


def main():
    # Print format to 3 decimal spaces and fix seed
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    # Fixed matrix parameters
    args.n = 20
    args.m = 50
    args.matrix_shape = (args.n, args.m)

    # Fixed noise distribution parameters
    args.sigma = 0.11

    # Generate random matrix
    A = generate_random_matrix(args.matrix_shape)

    # Calculate QR factorization of A
    (Q, _) = linalg.qr(A)

    # Set list of max eigenvalues
    max_eigenvalues = np.arange(1, args.max_eigenvalue, args.step)

    # Compare different execuutions of Cholesky
    times, errors, conditions = compare_choklesky(Q, max_eigenvalues)

    # Plot errors
    labels = ["Error Cholesky B", "Error Cholesky Be",
              "Error Scipy B", "Error Scipy Be"]
    plot_errors(x_range=max_eigenvalues, errors=errors, labels=labels)

    # Plot times
    labels = ["Time Cholesky B", "Time Cholesky Be",
              "Time Scipy B", "Time Scipy Be"]
    plot_times(x_range=max_eigenvalues, times=times, labels=labels)

    # Plot condition number
    # Plot times
    labels = ["Condition B", "Condition Be"]
    plot_condition(x_range=max_eigenvalues,
                   conditions=conditions, labels=labels)


if __name__ == "__main__":
    main()
