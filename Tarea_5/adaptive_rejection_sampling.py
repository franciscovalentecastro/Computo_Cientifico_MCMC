#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Parser arguments
parser = argparse.ArgumentParser(
    description='Adaptive rejection sampling.')
parser.add_argument('--seed', '--s',
                    type=int, default=1, metavar='N',
                    help='Seed to use in PRNG.')
parser.add_argument('--sample_size', '--size',
                    type=int, default=100, metavar='N',
                    help='The smaple size of "random" numbers.')
args = parser.parse_args()


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
    # Sample from piecewise exponential density pag. 61
    # Monte Carlo Statistical Methods - Robert, Casella

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
    points = [((y_value - b1) / a1, y_value)] + \
        points + \
        [((y_value - b2) / a2, y_value)]

    # Refine upper envelope to only lines
    for idx, p in enumerate(points):
        if idx <= 1 or idx == len(points) - 1:
            # Add inital and ending points
            envelope.append(p)
        elif idx > 1 and idx < len(points) - 1:
            # Find intersection of two lines to make refinement
            coef_1 = line_coef(points[idx - 2], points[idx - 1])
            coef_2 = line_coef(p, points[idx + 1])
            intersection = line_intersection(coef_1, coef_2)

            # Add refinement
            envelope.append(intersection)
            envelope.append(p)

    return envelope


def sample_from_upper_adaptive_envelope(points):
    # Sample from piecewise exponential density pag. 71 Ex. 2.39
    # Monte Carlo Statistical Methods - Robert, Casella

    # Common length
    n = len(points)

    # Construct piecewise envelope
    C = [0] * n  # (a,b) coefficients
    W = [0] * n  # scalar weights

    # Integrate picewise density
    for idx, p in enumerate(points):
        if idx > 0:
            # Line equation from piece
            x1, y1 = points[idx - 1]
            x2, y2 = points[idx]
            a, b = C[idx - 1] = line_coef(points[idx - 1], points[idx])

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


def adaptive_rejection_sampling(density, points, sample_size, seed=0):
    # Adaptive Rejection Sampling Algorithm pag. 57
    # Monte Carlo Statistical Methods - Robert, Casella

    # Points for the envelope
    S = points
    print('Initial number of points in envelope : #S = %d' % len(S))

    # Sample array
    sample = []
    y_sample = []

    # Count statistics
    accepted = 0
    refined = 0
    rejected = 0

    # Plot initial envelopes
    plot_envelopes(density, S, points_of_upper_adaptive_envelope(S),
                   filename="env_ini_%d.png" % args.sample_size)

    # Accept #sample_size samples
    while(len(sample) < sample_size):
        # Create lower envelope
        low_env = S

        def lower(x):
            return envelope(x, low_env)

        # Create upper envelope
        up_env = points_of_upper_adaptive_envelope(S)

        def upper(x):
            return envelope(x, up_env)

        # Sample from g_n distribution
        g_sample = sample_from_upper_adaptive_envelope(up_env)

        # Uniform (0, 1)
        rnd = np.random.rand()
        if rnd <= lower(g_sample) / upper(g_sample):  # Accept
            # Accept sample
            sample.append(g_sample)
            y_sample.append(rnd * upper(g_sample))

            # Sum an accepted sample
            accepted += 1
        elif rnd <= density(g_sample) / upper(g_sample):  # Accept and refine
            # Accept sample
            sample.append(g_sample)
            y_sample.append(rnd * upper(g_sample))

            # Refine the envelope
            S.append((g_sample, np.log(density(g_sample))))
            S.sort()

            # Sum an accepted + refining sample
            accepted += 1
            refined += 1
        else:
            # Reject sample
            rejected += 1

    # Number of "refinement" points in the envelope
    print('Number of points in envelope : #S = %d' % len(S))

    # Number of sample accpeted and rejected
    print('# Accepted = %d, # Rejected = %d, # Refined = %d' %
          (accepted, rejected, refined))

    # Plot last envelope and sample
    plot_envelopes(density, S, points_of_upper_adaptive_envelope(S),
                   sample=(sample, y_sample),
                   filename="env_sample_%d.png" % args.sample_size)

    return sample


def plot_envelopes(density, low_env, up_env, sample=([], []),
                   domain=(0, 10), filename='envelope.png'):
    def lower(x):
        return envelope(x, low_env)

    def upper(x):
        return envelope(x, up_env)

    # Slice the domain
    start, end = domain
    x_range = np.linspace(start, end, 1000)

    # Evaluate functions
    y_density = [density(x) for x in x_range]
    y_lower = [lower(x) for x in x_range]
    y_upper = [upper(x) for x in x_range]

    # Plot discretisation
    plt.plot(x_range, y_density, label='density')
    plt.plot(x_range, y_upper, label='upper envelope')
    plt.plot(x_range, y_lower, label='lower envelope')
    plt.scatter(sample[0], sample[1])
    plt.legend()
    plt.title('ARS enevelope of Gamma(2,1) '
              '(#Envelope Points = {}, N = {})'
              .format(len(low_env), args.sample_size))
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.show()


def main():
    # Print format to 3 decimal spaces and fix seed
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    # Distribution parameters
    alpha = 2
    beta = 1

    # Gamma density function
    def gamma_density(x):
        d = (beta ** alpha) / math.gamma(alpha)
        d *= x ** (alpha - 1)
        d *= math.exp(-beta * x)
        return d

    # Initial aproximations
    S = np.concatenate((np.arange(.1, 3, .4), np.arange(3, 5, 1.5)))
    print(S)
    S = [(s, np.log(gamma_density(s))) for s in S]

    # Adaptive rejection sampling
    adap = adaptive_rejection_sampling(gamma_density,
                                       S,
                                       args.sample_size,
                                       args.seed)

    # the histogram of the data
    n, bins, patches = plt.hist(adap,
                                'fd',
                                density=1,
                                facecolor='blue',
                                alpha=0.5)

    # Plot density
    x_range = np.linspace(0, 10, 100)
    y_density = [gamma_density(x) for x in x_range]
    plt.plot(x_range, y_density)

    # Format plot
    plt.title('ARS sample of Gamma(2,1) '
              '({} Bins, N = {})'.format(len(bins), args.sample_size))
    plt.savefig("histogram_sample_%d.png" % args.sample_size,
                bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == "__main__":
    main()
