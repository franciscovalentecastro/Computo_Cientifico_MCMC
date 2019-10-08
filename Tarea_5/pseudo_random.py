#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Constant 
maxInt = 2147483647  # 2^31 - 1

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
    div = maxInt  # 2^31 - 1

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
    a1, b1 = coef_1
    a2, b2 = coef_2

    # Caculate intersection
    x = (b2 - b1) / (a1 - a2)
    y = a1 * x + b1

    return (x, y)


def envelope(x, points):
    # Number of points
    n = len(points)

    # Find position of x in the array
    x_position = 0
    while x_position < n and x > points[x_position][0]:
        x_position += 1

    # Evaluate point
    if x_position == 0 or x_position == n:
        value = float('-inf')
    else:
        a, b = line_coef(points[x_position - 1], points[x_position])
        value = a * x + b

    return np.exp(value)



def points_of_upper_adaptive_envelope(points):
    # Points on upper envelope
    envelope = []

    # Slope of first and last piece 
    slope = .1
    y_value = -10

    # Calculate initial piece 
    x1, y1 = points[0]
    a1, b1 = (slope, y1 - slope * x1)

    # Calculate last piece
    x2, y2 = points[-1]
    a2, b2 = (-slope, y2 + slope * x2)

    # Add points in extremes
    points = [((y_value - b1)/a1, y_value)] + points + [((y_value - b2)/a2, y_value)]

    for idx, p in enumerate(points):
        if idx <= 1 or idx == len(points) - 1:
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
    n = len(points)

    # Construct piecewise envelope
    C = [0] * n  # (a,b)
    W = [0] * n  # scalar

    # Integrate picewise density
    for idx, p in enumerate(points):
        if idx > 0:
            # Line equation from piece
            x1, y1 = points[idx - 1]
            x2, y2 = points[idx]
            a, b = C[idx  - 1] = line_coef(points[idx - 1], points[idx])

            # Density weight of piece
            W[idx - 1] = np.exp(b) * (np.exp(a * x2) - np.exp(a * x1)) / a

    # Sum all weights and transform W to density
    weight = np.sum(W)
    W = W / weight

    # Pick random piece
    rnd = np.random.rand()
    x_position = sigma = 0
    while x_position - 1 < n and rnd > sigma + W[x_position]:
        sigma += W[x_position]
        x_position += 1

    # Use density on piece
    rnd = np.random.rand()
    x1, y1 = points[x_position]
    x2, y2 = points[x_position + 1]
    a, b = C[x_position]
    sample = np.log(np.exp(a * x1) +
                    rnd * (np.exp(a * x2) - np.exp(a * x1))) / a

    return sample


def plot_envelopes(density, low_env, up_env, sample, y_sample):
    def lower(x):
        return envelope(x, low_env)
    def upper(x):
        return envelope(x, up_env)

    x_range = np.linspace(0,10,100)
    y_density = [density(x) for x in x_range]
    y_lower = [lower(x) for x in x_range]
    y_upper = [upper(x) for x in x_range]

    plt.plot(x_range, y_density)
    plt.plot(x_range, y_upper)
    plt.plot(x_range, y_lower)
    plt.scatter(sample, y_sample)
    plt.show()

def adaptive_rejection_sampling(density, points, sample_size, seed=0):
    # Points for the envelope
    S = points

    # Sample array
    sample = []
    y_sample = []

    plot_envelopes(density, S,
                   points_of_upper_adaptive_envelope(S),
                   sample,y_sample)


    # Accept #sample_size samples
    while( len(sample) < sample_size ):

        low_env = S
        def lower(x):
            return envelope(x, low_env)

        up_env = points_of_upper_adaptive_envelope(S)
        def upper(x):
            return envelope(x, up_env)

        # Sample from g_n distribution
        g_sample = sample_from_upper_adaptive_envelope(up_env)

        rnd = np.random.rand()
        if rnd <= lower(g_sample) / upper(g_sample): # Accept
            # Accept sample
            sample.append(g_sample)
            y_sample.append(rnd * upper(g_sample))
        elif rnd <= density(g_sample) / upper(g_sample): # Accept and refine
            # Accept sample
            sample.append(g_sample)
            y_sample.append(rnd * upper(g_sample))
            
            # Refine the envelope
            S.append((g_sample, np.log(density(g_sample))))
            S.sort()

    plot_envelopes(density, S, 
                   points_of_upper_adaptive_envelope(S),
                   sample, y_sample)

    return sample


def uniform_sampling(sample_size, seed=1, a=0, b=1):
    # Maximum number
    div = maxInt  # 2^31 - 1

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
    S = np.linspace(0.1,5,10)
    S = [(s, np.log(gamma_density(s))) for s in S]
    adap = adaptive_rejection_sampling(gamma_density,
                                       S,
                                       args.sample_size,
                                       args.seed)

    num_bins = 80
    # the histogram of the data
    n, bins, patches = plt.hist(adap,
                                num_bins,
                                density=1,
                                facecolor='blue',
                                alpha=0.5)
    x_range = np.linspace(0,10,100)
    y_density = [gamma_density(x) for x in x_range]

    plt.plot(x_range, y_density)
    plt.show()

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



if __name__ == "__main__":
    main()
