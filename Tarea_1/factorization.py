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


def factorize(matrix):
    # Rename matrix
    A = matrix
    print("A :", A, sep="\n")

    print("LUP : ", end="\n")
    (L, U, P) = lup_factorization(A)

    # If LUP factorization was returned.
    if type(L) is np.ndarray:
        print("L : ", L, sep="\n")
        print("U : ", U, sep="\n")
        print("P : ", P, sep="\n")
        print("LUP factorization correct ? : ",
              np.allclose(np.transpose(P) @ L @ U, A), end="\n\n")

    print("LU : ", end="\n")
    (L, U) = lu_factorization(A)

    # If LU factorization was returned.
    if type(L) is np.ndarray:
        print("L :", L, sep="\n")
        print("U :", U, sep="\n")
        print("LU factorization correct ? : ",
              np.allclose(L @ U, A), end="\n\n")

    print("Cholesky : ", end="\n")
    R = cholesky(A)

    # If Cholesky factorization was returned.
    if type(R) is np.ndarray:
        print("R :", R, sep="\n")
        print("Cholesky factorization correct ? : ",
              np.allclose(np.transpose(R) @ R, A), end="\n\n")


def main():
    # Get seed parameter
    seed = 0 if len(sys.argv) == 1 else int(sys.argv[1])

    # Print format to 3 decimal spaces and fix seed
    np.random.seed(seed)
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    # Create matrix A
    A = np.matrix([[1, 0, 0, 0, 1],
                   [-1, 1, 0, 0, 1],
                   [-1, -1, 1, 0, 1],
                   [-1, -1, -1, 1, 1],
                   [-1, -1, -1, -1, 1]])

    # Try to factorize matrix A
    factorize(A)

    # Generate random matrix U(0,1)
    B = generate_random_matrix((5, 5))

    # Try to factorize matrix B
    factorize(B)


if __name__ == "__main__":
    main()
