#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from substitution import *
from factorization import *


def solve_system(A, b):
    # Get LUP decomposition
    (L, U, P) = lup_factorization(A)

    # Solve system Ly = Pb with y = Ux
    y = forward_substitution(L, P @ b)

    # Solve system Ux = y
    x = backward_substitution(U, y)

    return x


def main():
    # LUP to solve Ax = b system
    # Create the matrix
    A = np.matrix([[1, 0, 0, 0, 1],
                   [-1, 1, 0, 0, 1],
                   [-1, -1, 1, 0, 1],
                   [-1, -1, -1, 1, 1],
                   [-1, -1, -1, -1, 1]])

    # Create vector
    b = np.ones(5)

    # Solve system Ax = b
    x = solve_system(A, b)

    print("A = ", A)
    print("b = ", b)
    print("x = ", x)
    print("Ax = ", A @ x)
    print("Solution for Ax = b correct ? : ",
          np.allclose(A @ x, b),
          end="\n\n")

    # Solve Ax = b for random matrices
    for repetition in range(5):
        # Generate random matrix with uniform (0,1) elements
        A = generate_random_matrix((5, 5))

        # Solve system
        x = solve_system(A, b)

        print("A = ", A)
        print("b = ", b)
        print("x = ", x)
        print("Ax = ", A @ x)
        print("Solution for Ax = b correct ? : ",
              np.allclose(A @ x, b),
              end="\n\n")


if __name__ == "__main__":
    main()
