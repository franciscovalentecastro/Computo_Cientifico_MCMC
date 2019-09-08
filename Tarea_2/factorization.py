#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import numpy as np
import math
from substitution import *


def lu_factorization(matrix):
    # Get dimensions
    (n, m) = matrix.shape

    # Check if it is a square matrix
    if n != m:
        print("Not a square matrix.")
        return (-1, -1)

    # Check if there are zeroes in the diagonal
    if len(np.diag(matrix)) - np.count_nonzero(np.diag(matrix)) > 0:
        print("There are zeroes in the diagonal. Try LUP factorization.")
        return (-1, -1)

    # Rename and create matrices
    U = matrix.astype(np.float)
    L = np.identity(n, dtype=np.float)

    # Gaussian elimination without pivoting
    for kdx in range(n - 1):
        for jdx in range(kdx + 1, n):
            L[jdx, kdx] = U[jdx, kdx] / U[kdx, kdx]
            U[jdx, kdx:] -= U[kdx, kdx:] * L[jdx, kdx]

    return (L, U)


def lup_factorization(matrix):
    # Get dimensions
    (n, m) = matrix.shape

    # Check if it is a square matrix
    if n != m:
        print("Not a square matrix.")
        return (-1, -1, -1)

    # Rename and create matrices
    U = matrix.astype(np.float)
    L = np.identity(n, dtype=np.float)
    P = np.identity(n, dtype=np.float)

    # Gaussian elimination with partial pivoting.
    for kdx in range(0, n - 1):

        # Check if there are valid pivots.
        if len(U[kdx:, kdx]) - np.count_nonzero(U[kdx:, kdx]) > 0:
            print("Can't find a valid pivot.")
            return (-1, -1, -1)

        # Find pivot.
        idx_max = kdx + np.argmax(np.absolute(U[kdx:, kdx]))

        # Rearrange matrix rows to place the pivot.
        U[[kdx, idx_max]] = U[[idx_max, kdx]]
        P[[kdx, idx_max]] = P[[idx_max, kdx]]
        for jdx in range(0, kdx):
            L[kdx, jdx], L[idx_max, jdx] = L[idx_max, jdx], L[kdx, jdx]

        # Use pivot row to transform vectors below.
        for jdx in range(kdx + 1, n):
            L[jdx, kdx] = U[jdx, kdx] / U[kdx, kdx]
            U[jdx, kdx:] -= U[kdx, kdx:] * L[jdx, kdx]

    return (L, U, P)


def cholesky(matrix):
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

        R[kdx, kdx:] /= math.sqrt(R[kdx, kdx])

    return R


def qr_factorization(matrix):
    # Get dimensions
    (m, n) = matrix.shape

    # Rename and create matrices
    R = np.zeros((n, n), dtype=np.float)
    Q = matrix.astype(np.float)

    # Modified Gram-Schmidt (Trefethen pag. 58 )
    for idx in range(0, n):
        # Normalize ortogonal vector
        R[idx, idx] = math.sqrt(Q[:, idx].dot(Q[:, idx]))
        Q[:, idx] /= R[idx, idx]

        # Substract projection to q_i ortonormal vector
        for jdx in range(idx + 1, n):
            R[idx, jdx] = Q[:, idx].dot(Q[:, jdx])
            Q[:, jdx] -= R[idx, jdx] * Q[:, idx]

    return (Q, R)


def main():
    # Get seed parameter
    seed = 0 if len(sys.argv) == 1 else int(sys.argv[1])

    # Print format to 3 decimal spaces and fix seed
    np.random.seed(seed)
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    # Generate random matrix U(0,1)
    B = generate_random_matrix((5, 5))

    # QR factorization
    (Q, R) = qr_factorization(B)

    print(B)
    print(Q)
    print(R)
    print(Q @ R)

    print("QR factorization correct ? : ",
          np.allclose(Q @ R, B), end="\n\n")


if __name__ == "__main__":
    main()
