#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Constants
maxInt = 2147483647  # 2^31 - 1

# Parser arguments
parser = argparse.ArgumentParser(
    description='Linear congruential random number generator.')
parser.add_argument('--seed', '--s',
                    type=int, default=1, metavar='N',
                    help='Seed to use in PRNG.')
parser.add_argument('--sample_size', '--size',
                    type=int, default=100, metavar='N',
                    help='The smaple size of "random" numbers.')
args = parser.parse_args()


def linear_congurential_generator(sample_size, seed=1):
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


def uniform_sampling(sample_size, seed=1, a=0, b=1):
    # Maximum number
    div = maxInt  # 2^31 - 1

    # Obtain uniform sample in the range (0,1)
    unif = linear_congurential_generator(sample_size, seed) / div

    # Transform to range (a,b)
    return ((b - a) * unif) + a


def exponential_sampling(sample_size, seed=1, lmbd=1):
    unif = uniform_sampling(sample_size, seed)
    return - np.log(1 - unif) / lmbd


def main():
    # Print format to 3 decimal spaces and fix seed
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    # Uniform sampling
    unif_01 = uniform_sampling(args.sample_size, args.seed, a=0, b=1)

    # the histogram of the generated sample
    n, bins, patches = plt.hist(unif_01,
                                'fd',
                                density=1,
                                facecolor='blue',
                                alpha=0.5)
    plt.title('Linear Congruential Generator '
              '(Bin size {}, N = {})'.format(len(bins), args.sample_size))
    plt.savefig("histogram_unif_%d.png" % args.sample_size,
                bbox_inches='tight', pad_inches=0)
    plt.show()

    # qqplot of the generated sample
    stats.probplot(unif_01, dist="uniform", plot=plt)
    plt.title("Uniform Q-Q plot (N = {})".format(args.sample_size))
    plt.savefig("qqplot_unif_%d.png" % args.sample_size,
                bbox_inches='tight', pad_inches=0)
    plt.show()

    # Kolmogorov Smirnov test of uniformity
    print('Kolmogorov-Smirnov test of uniformity :')
    print(stats.kstest(unif_01, 'uniform'))

    # --- #

    # Exponential sampling
    exp_1 = exponential_sampling(args.sample_size, args.seed, lmbd=2)
    exp_2 = exponential_sampling(args.sample_size, args.seed, lmbd=4)
    exp_3 = exponential_sampling(args.sample_size, args.seed, lmbd=8)

    # the histogram of the data
    n, bins, patches = plt.hist(exp_1, 'fd', alpha=0.5)
    n, bins, patches = plt.hist(exp_2, 'fd', alpha=0.5)
    n, bins, patches = plt.hist(exp_3, 'fd', alpha=0.5)
    labels = ['lambda = 2', 'lambda = 4', 'lambda = 8']
    plt.legend(labels=labels)
    plt.title('Sampling Exponential Distribution '
              'from LCG (N = {})'.format(args.sample_size))
    plt.savefig("histogram_exp_%d.png" % args.sample_size,
                bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == "__main__":
    main()
