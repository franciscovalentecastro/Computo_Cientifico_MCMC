#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
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
    # Get seed parameter
    seed = 0 if len(sys.argv) == 1 else int(sys.argv[1])

    # Print format to 3 decimal spaces and fix seed
    np.random.seed(seed)
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    # Create matrix A (same as in factorization.py)
    A = np.matrix([[1, 0, 0, 0, 1],
                   [-1, 1, 0, 0, 1],
                   [-1, -1, 1, 0, 1],
                   [-1, -1, -1, 1, 1],
                   [-1, -1, -1, -1, 1]])

    # Generate random matrix U(0,1) (same as in factorization.py)
    B = generate_random_matrix((5, 5))

    # Solve systems Ax = b and Bx = b, for random b vectors.
    for repetition in range(5):
        # Generate random vector b with uniform (0,1) elements
        b = generate_random_matrix((5,))

        # Solve system Ax = b
        x = solve_system(A, b)

        # Print solution.
        print("A : ", A, sep="\n")
        print("b : ", b)
        print("x : ", x)
        print("Ax : ", A @ x)
        print("Solution for Ax = b correct ? : ",
              np.allclose(A @ x, b),
              end="\n\n")

        # Solve system Bx = b
        x = solve_system(B, b)

        # Print solution.
        print("B : ", B, sep="\n")
        print("b : ", b)
        print("x : ", x)
        print("Bx : ", B @ x)
        print("Solution for Bx = b correct ? : ",
              np.allclose(B @ x, b),
              end="\n\n")


if __name__ == "__main__":
    main()
