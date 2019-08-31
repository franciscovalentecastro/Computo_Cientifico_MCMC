#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


def backward_substitution(upper_triangular_matrix, constant_vector):
    # Rename matrix and vector
    A = upper_triangular_matrix
    b = constant_vector

    # Get dimensions
    (n, m) = A.shape
    (k,) = b.shape

    # Check if it is a square matrix and dimensions match
    if n != m:
        print("Not a square matrix.")
        return -1

    if n != k:
        print("Dimensions don't match.")
        return -1

    # Check if matrix is upper triangular
    if not np.allclose(A, np.triu(A)):
        print("Not an upper triangular matrix.")
        return -1

    # Check if there are zeroes in the diagonal
    if len(np.diag(A)) - np.count_nonzero(np.diag(A)) > 0:
        print("There is not an unique solution for the system.")
        return -1

    # Backward substitution algorithm
    x = np.zeros(n)
    for idx in range(n - 1, -1, -1):
        x[idx] = b[idx]
        for jdx in range(idx + 1, n):
            x[idx] -= x[jdx] * A[idx, jdx]
        x[idx] /= A[idx, idx]

    return x


def forward_substitution(lower_triangular_matrix, constant_vector):
    # Rename matrix and vector
    A = lower_triangular_matrix
    b = constant_vector

    # Get dimensions
    (n, m) = A.shape
    (k,) = b.shape

    # Check if it is a square matrix and dimensions match
    if n != m:
        print("Not a square matrix.")
        return -1

    if n != k:
        print("Dimensions don't match.")
        return -1

    # Check if matrix is lower triangular
    if not np.allclose(A, np.tril(A)):
        print("Not an lower triangular matrix.")
        return -1

    # Check if there are zeroes in the diagonal
    if len(np.diag(A)) - np.count_nonzero(np.diag(A)) > 0:
        print("There is not an unique solution for the system.")
        return -1

    # Forward substitution algorithm
    x = np.zeros(n)
    for idx in range(0, n):
        x[idx] = b[idx]
        for jdx in range(0, idx):
            x[idx] -= x[jdx] * A[idx, jdx]
        x[idx] /= A[idx, idx]

    return x


def generate_random_matrix(matrix_shape):
    return np.random.rand(*matrix_shape)


def main():
    # Generate random (upper triangular) system of equations
    A = np.triu(generate_random_matrix((5, 5)))
    b = generate_random_matrix((5,))
    print(b)

    print("A = ", A, end="\n\n")
    print("b = ", b, end="\n\n")

    # Solve by backward substitution
    x = backward_substitution(A, b)

    if type(x) is np.ndarray:
        print("x = ", x, end="\n\n")
        print("Solution for Ax = b correct ? : ",
              np.allclose(A @ x, b), end="\n\n")

    # Generate random (lower triangular) system of equations
    A = np.tril(generate_random_matrix((5, 5)))
    b = generate_random_matrix((5,))

    print("A = ", A, end="\n\n")
    print("b = ", b, end="\n\n")

    # Solve by backward substitution
    x = forward_substitution(A, b)

    if type(x) is np.ndarray:
        print("x = ", x, end="\n\n")
        print("Solution for Ax = b correct ? : ",
              np.allclose(A @ x, b), end="\n\n")


if __name__ == "__main__":
    main()
