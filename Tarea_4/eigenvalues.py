#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import numpy as np

from factorization import *

# Parser arguments
parser = argparse.ArgumentParser(
    description='Calculate Eigenvalues with QR Algorithm')
parser.add_argument('--plot', '--p',
                    action='store_true',
                    help='plot dataset sample')
args = parser.parse_args()


def hessenberg_form(matrix):
    # Get dimensions
    (m, n) = matrix.shape

    # Rename and matrix
    A = np.array(matrix, dtype=np.float)

    # Check if matrix is diagonal
    if np.count_nonzero(A - np.diag(np.diagonal(A))) == 0:
        print("The matrix is diagonal.")
        return A

    for kdx in range(0, m - 2):
        # Pick column to reduce
        x = A[kdx + 1:m, kdx]

        # No need to reduce
        if np.count_nonzero(x) == 0:
            continue

        # Householder refection vector
        v = (np.sign(x[0]) *
             np.linalg.norm(x) *
             np.eye(1, m - kdx - 1, 0)) + x

        # Normalize vector
        v = v / np.linalg.norm(v)

        # Transform matrices Qt * A * Q
        A[kdx + 1:m, kdx:m] -= 2.0 * v.T @ (v @ A[kdx + 1:m, kdx:m])
        A[0:m, kdx + 1:m] -= 2.0 * (A[0:m, kdx + 1:m] @ v.T) @ v

    return A


def qr_algorithm(matrix, iterations=1000):
    # Get dimensions
    (m, n) = matrix.shape

    # Check if it is a square matrix
    if n != m:
        print("Not a square matrix.")
        return -1

    # Rename and matrix
    A = np.array(matrix, dtype=np.float)
    Q_k = np.identity(m)

    # To Hessenberg form
    A_k = hessenberg_form(A)

    for kdx in range(0, iterations):
        # Pick a shift
        mu = A_k[m - 1, m - 1]

        # Calculate QR factorization
        (Q, R) = qr_factorization(A_k - mu * np.identity(m))
        # (Q, R) = np.linalg.qr(A_k - mu * np.identity(m))

        # Multiply by Q from the right
        A_k = R @ Q + mu * np.identity(m)

        # Calculate eigenvectors
        Q_k = Q_k @ Q

    return (A_k, Q_k)


def main():
    # Print format to 3 decimal spaces and fix seed
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    # Calculate eigenvalues of symmetric matrix
    for exponent in range(1, 6):
        # Create epsilon
        epsilon = 10 ** exponent

        # Create matrix
        A = np.array([[8, 1, 0],
                      [1, 4, epsilon],
                      [0, epsilon, 1]])

        # Calculate eigenvalues
        (A_k, Q_k) = qr_algorithm(A, iterations=epsilon)

        print("epsilon : ", epsilon)
        print("A : ", A, sep="\n")

        # Sort eigenvalues
        sortedIdx = np.argsort(np.diag(A_k))

        # Print sorted eigenvalues and eigenvectors
        print("eigenvalues : ", np.diag(A_k)[sortedIdx], end="\n")
        print("eigenvectors : ", Q_k[:, sortedIdx], sep="\n", end="\n\n")


if __name__ == "__main__":
    main()
