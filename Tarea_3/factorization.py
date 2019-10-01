#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

from substitution import *


def cholesky_factorization(matrix):
    # Get dimensions
    (n, m) = matrix.shape

    # Check if it is a square matrix
    if n != m:
        print("Not a square matrix.")
        return (-1,)

    if not np.allclose(matrix, matrix.T):
        print("Not a symmetric matrix.")
        return (-1,)

    # Check if there are zeroes in the diagonal
    if len(np.diag(matrix)) - np.count_nonzero(np.diag(matrix)) > 0:
        print("Not an hermitian positive definite matrix.")
        return (-1,)

    # Rename and copy matrix
    R = np.triu(matrix.astype(np.float))

    # Cholesky factoization (Trefethen pag. 175)
    for kdx in range(0, n):
        for jdx in range(kdx + 1, n):
            R[jdx, jdx:] -= R[kdx, jdx:] * (R[kdx, jdx] / R[kdx, kdx])

        if R[kdx, kdx] < 0:
            print("Not an hermitian positive definite matrix.")
            return (-1,)

        R[kdx, kdx:] /= np.sqrt(R[kdx, kdx])

    return R


def qr_factorization(matrix):
    # Get dimensions
    (m, n) = matrix.shape

    # Rename and create matrices
    R = np.zeros((n, n), dtype=np.float)
    Q = np.array(matrix, dtype=np.float)

    # Modified Gram-Schmidt (Trefethen pag. 58 )
    for idx in range(0, n):
        # Normalize ortogonal vector
        R[idx, idx] = np.sqrt(Q[:, idx].dot(Q[:, idx]))
        Q[:, idx] /= R[idx, idx]

        # Substract projection to q_i ortonormal vector
        for jdx in range(idx + 1, n):
            R[idx, jdx] = Q[:, idx].dot(Q[:, jdx])
            Q[:, jdx] -= R[idx, jdx] * Q[:, idx]

    return (Q, R)


def main():
    # Print format to 3 decimal spaces and fix seed
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    # Generate random matrix U(0,1)
    A = generate_random_matrix((5, 5))

    # QR factorization
    (Q, R) = qr_factorization(A)

    print("A : ", A, sep="\n")
    print("Q : ", Q, sep="\n")
    print("R : ", R, sep="\n")
    print("Q * R : ", Q @ R, sep="\n")

    print("QR factorization correct ? : ",
          np.allclose(Q @ R, A), end="\n\n")


if __name__ == "__main__":
    main()
