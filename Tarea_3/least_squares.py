#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import numpy as np
from scipy import linalg

import matplotlib.pyplot as plt

from factorization import *

# Parser arguments
parser = argparse.ArgumentParser(description='Compare Cholesky Factorization')
parser.add_argument('--iterations', '--it',
                    type=int, default=10, metavar='N',
                    help='Number of iterations (default: 10)')
parser.add_argument('--colinear', '--c',
                    type=int, default=0, metavar='N',
                    help='Number of colinear columns to set (default: 0)')
args = parser.parse_args()


def least_squares_estimator(X, Y):
    # Get QR decomposition of data matrix
    (Q, R) = qr_factorization(X)

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

    # Solve system R * beta = y_prime
    beta = Xt_X_inv @ X.T @ Y

    return beta


def generate_linear_data(X, beta, size, sigma):
    # Create data vectors
    Y = np.random.normal(loc=0.0, scale=sigma, size=size)

    # Add linear data
    Y += X @ beta

    return Y


def compare_least_squares(X, Y):
    # Add noise delta to matrix
    X_delta = X + np.random.normal(loc=0.0,
                                   scale=args.sigma_delta,
                                   size=args.dimension)

    # Calculate condition number
    cond_X = np.linalg.cond(X)
    cond_delta_X = np.linalg.cond(X_delta)
    conditions = [cond_X, cond_delta_X]

    # Get least squares estimator
    b1 = least_squares_estimator(X, Y)
    b2 = least_squares_estimator(X_delta, Y)
    b3 = least_squares_estimator_scipy_inv(X_delta, Y)
    b4 = least_squares_estimator_scipy_qr(X_delta, Y)

    # Calculate estimation errors
    error_b1 = np.linalg.norm(args.beta - b1)
    error_b2 = np.linalg.norm(args.beta - b2)
    error_b3 = np.linalg.norm(args.beta - b3)
    error_b4 = np.linalg.norm(args.beta - b4)
    errors = [error_b1, error_b2, error_b3, error_b4]

    # Print intial matrices
    print("Conditition X : ", cond_X)
    print("Conditition X + delta_X : ", cond_delta_X)

    # Print the resulting estimations
    print(args.beta, ' - Real Beta')
    print('%s - Error : %.3f'
          ' - Least Square Estimator Own QR with X'
          % (str(b1), error_b1))
    print('%s - Error : %.3f'
          ' - Least Square Estimator Own QR with X + delta_X'
          % (str(b2), error_b2))
    print('%s - Error : %.3f'
          ' - Least Square Estimator Scipy Inverse with X + delta_X'
          % (str(b3), error_b3))
    print('%s - Error : %.3f'
          ' - Least Square Estimator Scipy QR with X + delta_X'
          % (str(b4), error_b4), end='\n\n')

    return (errors, conditions)


def plot_executions(execution_sample):
    # Generate ex to plot
    eb1 = [[], [], [], [], []]
    eb2 = [[], [], [], [], []]
    eb3 = [[], [], [], [], []]
    eb4 = [[], [], [], [], []]
    cx = [[], [], [], [], []]
    cdx = [[], [], [], [], []]

    # Array of means
    meb1 = []
    meb2 = []
    meb3 = []
    meb4 = []
    mcx = []
    mcdx = []

    # Array of variance
    veb1 = []
    veb2 = []
    veb3 = []
    veb4 = []
    vcx = []
    vcdx = []

    # Reshape sample
    for element in execution_sample:
        eb1[element[0]].append(element[1][0])
        eb2[element[0]].append(element[1][1])
        eb3[element[0]].append(element[1][2])
        eb4[element[0]].append(element[1][3])
        cx[element[0]].append(element[2][0])
        cdx[element[0]].append(element[2][1])

    # Range of plot
    x_range = range(min(args.colinear + 1, 5))

    # Calculate mean and variance
    for idx in x_range:
        meb1.append(np.mean(eb1[idx]))
        meb2.append(np.mean(eb2[idx]))
        meb3.append(np.mean(eb3[idx]))
        meb4.append(np.mean(eb4[idx]))
        mcx.append(np.mean(cx[idx]))
        mcdx.append(np.mean(cdx[idx]))
        veb1.append(np.var(eb1[idx]))
        veb2.append(np.var(eb2[idx]))
        veb3.append(np.var(eb3[idx]))
        veb4.append(np.var(eb4[idx]))
        vcx.append(np.var(cx[idx]))
        vcdx.append(np.var(cdx[idx]))

    # Plot mean
    plt.plot(x_range, meb1, ".-", label="Own QR with X", alpha=0.4)
    plt.plot(x_range, meb2, ".-", label="Own QR with X + D_X", alpha=0.4)
    plt.plot(x_range, meb3, ".-", label="Scipy Inv with X + D_X", alpha=0.4)
    plt.plot(x_range, meb4, ".-", label="Scipy QR with X + D_X", alpha=0.4)
    plt.xlabel("Number of Colinear Columns")
    plt.ylabel("Error")

    # Construct plot
    plt.legend(loc='upper right')
    plt.savefig('erros-N=%d' % len(x_range), bbox_inches='tight')
    plt.show()

    # Plot condition
    plt.plot(x_range, mcx, ".-", label="Condition of X", alpha=0.7)
    plt.plot(x_range, mcdx, ".-", label="Condition of X + D_X", alpha=0.7)
    plt.xlabel("Number of Colinear Columns")
    plt.ylabel("Mean of Condition Number")

    # Construct plot
    plt.legend(loc='upper right')
    plt.savefig('erros-N=%d' % len(x_range), bbox_inches='tight')
    plt.show()


def main():
    # Print format to 3 decimal spaces and fix seed
    np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})

    # Fixed problem parameters
    args.n = 20
    args.d = 5
    args.sigma = 0.15
    args.sigma_delta = .01
    args.beta = np.array([5, 4, 3, 2, 1]).T
    args.dimension = (args.n, args.d)

    # Sample of executions
    execution_sample = []

    for idx in range(args.iterations):
        for colinear in range(min(args.colinear + 1, 5)):
            # Generate a random matrix
            X = generate_random_matrix(args.dimension)

            # Set args.colinear columns to be colinear
            for idx in range(1, min(1 + colinear, 5)):
                X[:, idx] = (idx + 1) * X[:, 0] \
                    + np.random.normal(loc=0.0, scale=.01, size=args.n)

            # Generate linear data
            Y = generate_linear_data(X, args.beta, args.n, args.sigma)

            # Compare diferent least squares algorithms
            errors, conditions = compare_least_squares(X, Y)

            # Add a sample
            execution_sample.append((colinear, errors, conditions))

    # Plot executions
    plot_executions(execution_sample)


if __name__ == "__main__":
    main()
