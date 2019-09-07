#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import numpy as np
import timeit
import functools

import matplotlib.pyplot as plt

from substitution import *
from factorization import *


def measure_factorization_time(matrix, method, number=1, **kwargs):
    # Calculate sum of execution times of "number".
    # Return the mean of the execution time when dividing by "number".
    return timeit.timeit(functools.partial(method, matrix),
                         number=number, **kwargs) / number


def verify_factorization(matrix):
    # Rename matrix
    A = matrix.astype(np.float)

    # Matrix decomposition
    (L, U, P) = lup_factorization(A)
    R = cholesky(A)

    # Compare if factorization is equal to the original matrix.
    print("All close LUP : ", np.allclose(A, np.transpose(P) @ L @ U))
    print("All close Cholesky : ", np.allclose(np.transpose(R) @ R, A))
    print("All close between : ",
          np.allclose(np.transpose(R) @ R, np.transpose(P) @ L @ U))


def compare_execution_speed_different_size_matrices(max_size,
                                                    debug=True,
                                                    calculateStep=1,
                                                    printStep=50):

    # Sample of times
    times_cholesky = []
    times_lup = []

    # Time range
    times = range(calculateStep, max_size, calculateStep)

    # Loop trough matrix sizes
    for size in times:

        # Generate random positve symetric matrix
        A = generate_random_matrix((size, size))
        A = A @ np.transpose(A)

        # Verify if factorization was correct.
        if debug:
            verify_factorization(A)

        # Measure factorization execution time and save in sample.
        tCholesky = measure_factorization_time(A, cholesky)
        times_cholesky.append(tCholesky)

        tLUP = measure_factorization_time(A, lup_factorization)
        times_lup.append(tLUP)

        # Every printStep iterations show current time information.
        if size % printStep == 0:
            print("Matrix Size : " + str(size),
                  "Cholesky Time : %.5f" % tCholesky,
                  "LUP Time : %.5f" % tLUP, sep=', ')

    # Fix plot size
    plt.figure(figsize=(10, 10))

    # Set axis name
    plt.xlabel("Matrix Size")
    plt.ylabel("Execution Time")

    # Create plot and save to file.
    plt.plot(times, times_cholesky, label="Cholesky")
    plt.plot(times, times_lup, label="LUP")
    plt.legend(loc='upper right')
    plt.savefig("execution_times_graph_N=%d.png" % max_size,
                bbox_inches='tight')
    plt.show()


def compare_execution_speed_fixed_size_matrices(size,
                                                sample_size,
                                                debug=True,
                                                printStep=50):
    # Sample of times
    times_cholesky = []
    times_lup = []
    times_ratios = []

    for repetition in range(sample_size):
        # Generate random positve symetric matrix
        A = generate_random_matrix((size, size))
        A = A @ np.transpose(A)

        # Verify if factorization was correct.
        if debug:
            verify_factorization(A)

        # Measure factorization execution time and save in sample.
        tCholesky = measure_factorization_time(A, cholesky)
        times_cholesky.append(tCholesky)

        tLUP = measure_factorization_time(A, lup_factorization)
        times_lup.append(tLUP)

        # Calculate ratios of times.
        times_ratios.append(tCholesky / tLUP)

        # Every printStep iterations show current time information.
        if repetition % printStep == 0:
            print("Repetition : " + str(repetition),
                  "Cholesky Time : %.5f" % tCholesky,
                  "LUP Time : %.5f" % tLUP, sep=', ')

    # Plot and save histogram.
    plt.figure(figsize=(10, 10))
    plt.hist([times_cholesky, times_lup],
             bins='auto', label=['cholesky', 'lup'])
    plt.legend(loc='upper right')
    plt.savefig("execution_times_histogram_N=%d.png" % sample_size,
                bbox_inches='tight')
    plt.show()


def main():
    # Set up default parameters
    max_size = 500
    size = 15
    sample_size = 500
    debug = False

    # Receive command line parameters.
    if len(sys.argv) > 4:
        max_size = int(sys.argv[1])
        size = int(sys.argv[2])
        sample_size = int(sys.argv[3])
        debug = True if sys.argv[4] == "True" else False

    compare_execution_speed_different_size_matrices(max_size, debug)
    compare_execution_speed_fixed_size_matrices(size, sample_size, debug)


if __name__ == "__main__":
    main()
