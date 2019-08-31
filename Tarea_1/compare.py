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
    return timeit.timeit(functools.partial(method, matrix),
                         number=number, **kwargs) / number


def verify_factorization(matrix):
    # Rename matrix
    A = matrix.astype(np.float)

    # Matrix decomposition
    (L, U, P) = lup_factorization(A)
    R = cholesky(A)

    print("All close LUP : ", np.allclose(A, np.transpose(P) @ L @ U))
    print("All close Cholesky : ", np.allclose(np.transpose(R) @ R, A))
    print("All close between : ",
          np.allclose(np.transpose(R) @ R, np.transpose(P) @ L @ U))


def compare_execution_speed_different_sized_matrices(max_size,
                                                     debug=True,
                                                     printStep=50):

    # Sample of times
    times_cholesky = []
    times_lup = []

    # Loop trough matrix sizes
    for size in range(1, max_size + 1):

        # Generate random positve symetric matrix
        A = generate_random_matrix((size, size))
        A = A @ np.transpose(A)

        if debug:
            verify_factorization(A)

        tCholesky = measure_factorization_time(A, cholesky)
        times_cholesky.append(tCholesky)

        tLUP = measure_factorization_time(A, lup_factorization)
        times_lup.append(tLUP)

        if size % printStep == 0:
            print("Matrix Size : " + str(size),
                  "Cholesky Time : %.5f" % tCholesky,
                  "LUP Time : %.5f" % tLUP, sep=', ')

    plt.plot(range(1, max_size + 1), times_cholesky, label="Cholesky")
    plt.plot(range(1, max_size + 1), times_lup, label="LUP")
    plt.legend(loc='upper right')
    plt.savefig("execution_times_graph_N=%d.png" % max_size,
                bbox_inches='tight')
    plt.show()


def compare_execution_speed_fixed_sized_matrices(size,
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

        if debug:
            verify_factorization(A)

        tCholesky = measure_factorization_time(A, cholesky)
        times_cholesky.append(tCholesky)

        tLUP = measure_factorization_time(A, lup_factorization)
        times_lup.append(tLUP)

        times_ratios.append(tCholesky / tLUP)

        if repetition % printStep == 0:
            print("Repetition : " + str(repetition),
                  "Cholesky Time : %.5f" % tCholesky,
                  "LUP Time : %.5f" % tLUP, sep=', ')

    plt.hist([times_cholesky, times_lup],
             bins='auto', label=['cholesky', 'lup'])
    plt.legend(loc='upper right')
    plt.savefig("execution_times_histogram_N=%d.png" % sample_size,
                bbox_inches='tight')
    plt.show()


def main():

    # Get parameters
    if len(sys.argv) > 4:
        max_size = int(sys.argv[1])
        size = int(sys.argv[2])
        sample_size = int(sys.argv[3])
        debug = True if sys.argv[4] == "True" else False
    else:
        print("Not enough parameters")
        return

    compare_execution_speed_different_sized_matrices(max_size, debug)
    compare_execution_speed_fixed_sized_matrices(size, sample_size, debug)


if __name__ == "__main__":
    main()
