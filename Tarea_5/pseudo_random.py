#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Parser arguments
parser = argparse.ArgumentParser(
    description='Random number generator')
parser.add_argument('--seed', '--s',
                    type=int, default=1, metavar='N',
                    help='Seed to use in PRNG.')
parser.add_argument('--sample_size', '--size',
                    type=int, default=100, metavar='N',
                    help='The smaple size of "random" numbers.')
args = parser.parse_args()


def linear_congurent_generator(sample_size, seed=1):
    # Recurrence vector
    x = np.ones(5 + sample_size) * seed

    # Divisor for modulo operation
    div = 2147483647  # 2^31 - 1

    # Simulate for #sample_size of iterations
    for idx in range(5, sample_size + 5):
        # Linear recurrence
        x[idx] = 107374182 * x[idx - 1]
        x[idx] += 104420 * x[idx - 5]

        # Modulo operation
        x[idx] = np.mod(x[idx], div)

    return x[5:]


def line_coef(x_1, x_2):
    # Unpack values
    x1, y1 = x_1
    x2, y2 = x_2

    # Obtain y = a + b*x line equation
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1

    return (a, b)


def line_intersection(coef_1, coef_2):
    # Unpack coeficients
    b1, a1 = coef_1
    b2, a2 = coef_2

    # Caculate intersection
    x = (a1 - a2) / (b2 - b1)
    y = b1 * x + a1

    return (x, y)


def lower_adaptive_envelope(x, points):
    # Number of points
    n = len(points)

    # Find position of x in the array
    x_position = 0
    while x > points[x_position] and x_position < n:
        x_position += 1

    # Evaluate point
    if x_position == 0 or x_position == n:
        value = float('-inf')
    else:
        a, b = line_coef(x,
                         points[x_position - 1],
                         points[x_position])
        value = a * x + b

    return np.exp(value)


def uppper_adaptive_envelope(x, points):
    # Number of points
    n = len(points)

    # Find position of x in the array
    x_position = 0
    while x > points[x_position] and x_position < n:
        x_position += 1

    # Evaluate point
    if x_position == 0 or x_position == n:
        value = float('-inf')
    else:
        a, b = line_coef(x,
                         points[x_position - 1],
                         points[x_position])
        value = a * x + b

    return np.exp(value)


def points_of_upper_adaptive_envelope(points):
    # Points on upper envelope
    envelope = []

    # Add points in extremes
    points = [(float('-inf'), float('-inf'))] + points

    for idx, p in enumerate(points):
        if idx == 1:
            envelope.append(p)
        elif idx > 1 and idx < len(points) - 1:
            coef_1 = line_coef(points[idx - 2], points[idx - 1])
            coef_2 = line_coef(p, points[idx + 1])
            intersection = line_intersection(coef_1, coef_2)
            envelope.append(intersection)
            envelope.append(p)

    return envelope


def sample_from_upper_adaptive_envelope(points):
    # Common length
    n = len(points) + 2

    # Construct piecewise envelope
    C = [0] * n  # (a,b)
    W = [0] * n  # scalar

    # Integrate picewise density
    # First integral
    x, y = points[0]
    a, b = C[0] = (.001, y - (.001) * x)
    W[0] = np.exp(a * x + b) / a

    # Intermediate integral
    for idx, p in enumerate(points):
        if idx > 0:
            x1, y1 = points[idx - 1]
            x2, y2 = points[idx]

            a, b = C[idx] = line_coef(points[idx - 1],
                                      points[idx])
            W[idx] = np.exp(b) * (np.exp(a * x2) - np.exp(a * x1)) / a

    # Last integral
    x, y = points[-1]
    a, b = C[-1] = (-.001, y - (-.001) * x)
    W[-1] = -np.exp(a * x + b) / a

    # Complete integral
    weight = np.sum(W)
    W = W / weight

    print(W)

    # Pick random piece
    rnd = np.random.rand()
    x_position = sigma = 0
    while rnd > sigma + W[x_position] and x_position - 1 < n:
        x_position += 1
        sigma += W[x_position]

    # Use density on piece
    x1, y1 = points[x_position]
    x2, y2 = points[x_position + 1]
    a, b = C[x_position]
    sample = np.log(np.exp(a * x) +
                    rnd * (np.exp(a * x2) - np.exp(a * x1))) / a

    return sample


def adaptive_rejection_sampling(density, points, sample_size, seed=0):
    # Points for the envelope
    S = points

    # Sample array
    sample = []

    for idx in range(sample_size):
        def lower(x):
            return lower_adaptive_envelope(x, S)

        up_env = points_of_upper_adaptive_envelope(S)

        def upper(x):
            return uppper_adaptive_envelope(x, up_env)

        # Sample from g_n distribution
        g_sample = sample_from_upper_adaptive_envelope(S)

        rnd = np.random.rand()
        if rnd <= lower(g_sample) / upper(g_sample):
            sample.append(g_sample)
        elif rnd <= density(g_sample) / upper(g_sample):
            sample.append(g_sample)
            S.append(g_sample)
            S = sorted(S)

    return sample


def uniform_sampling(sample_size, seed=1, a=0, b=1):
    # Maximum number
    div = 2147483647  # 2^31 - 1

    # Obtain uniform sample in the range (0,1)
    unif = linear_congurent_generator(sample_size, seed) / div

    # Transform to range (a,b)
    return ((b - a) * unif) + a


def exponential_sampling(sample_size, seed=1, lmbd=1):
    unif = uniform_sampling(sample_size, seed)
    return - np.log(1 - unif) / lmbd


def main():
    # Print format to 3 decimal spaces and fix seed
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    alpha = 2
    beta = 1

    def gamma_density(x):
        d = (beta ** alpha) / math.gamma(alpha)
        d *= x ** (alpha - 1)
        d *= math.exp(-beta * x)
        return d

    # Adaptive rejection sampling
    S = [.1, .5, 1, 1.5, 10]
    S = [(s, gamma_density(s)) for s in S]
    adap = adaptive_rejection_sampling(gamma_density,
                                       S,
                                       args.sample_size,
                                       args.seed)
    print(adap)

    num_bins = 100
    # the histogram of the data
    n, bins, patches = plt.hist(adap,
                                num_bins,
                                density=1,
                                facecolor='blue',
                                alpha=0.5)
    plt.show()

    return

    unif_01 = uniform_sampling(args.sample_size, args.seed, a=0, b=1)
    unif_23 = uniform_sampling(args.sample_size, args.seed, a=2, b=3)
    unif_78 = uniform_sampling(args.sample_size, args.seed, a=3, b=8)

    # print(unif_01)
    # print(unif_23)

    unif = np.concatenate((unif_01, unif_23, unif_78))
    # print(unif)

    num_bins = 100
    # the histogram of the data
    n, bins, patches = plt.hist(unif,
                                num_bins,
                                density=1,
                                facecolor='blue',
                                alpha=0.5)
    plt.show()

    # --- #

    exp_1 = exponential_sampling(args.sample_size, args.seed, lmbd=1)
    exp_2 = exponential_sampling(args.sample_size, args.seed, lmbd=2)
    exp_3 = exponential_sampling(args.sample_size, args.seed, lmbd=3)
    # print(unif)

    num_bins = 100
    # the histogram of the data
    n, bins, patches = plt.hist(exp_1, num_bins, density=1, alpha=0.5)
    n, bins, patches = plt.hist(exp_2, num_bins, density=1, alpha=0.5)
    n, bins, patches = plt.hist(exp_3, num_bins, density=1, alpha=0.5)
    plt.show()

    # Calculate eigenvalues of symmetric matrix
    # for exponent in range(1, 6):
    #     # Create epsilon
    #     epsilon = 10 ** exponent

    #     # Create matrix
    #     A = np.array([[8, 1, 0],
    #                   [1, 4, epsilon],
    #                   [0, epsilon, 1]])

    #     # Calculate eigenvalues
    #     (A_k, Q_k) = qr_algorithm(A, iterations=epsilon)

    #     print("epsilon : ", epsilon)
    #     print("A : ", A, sep="\n")

    #     # Sort eigenvalues
    #     sortedIdx = np.argsort(np.diag(A_k))

    #     # Print sorted eigenvalues and eigenvectors
    #     print("eigenvalues : ", np.diag(A_k)[sortedIdx], end="\n")
    #     print("eigenvectors : ", Q_k[:, sortedIdx], sep="\n", end="\n\n")


if __name__ == "__main__":
    main()
