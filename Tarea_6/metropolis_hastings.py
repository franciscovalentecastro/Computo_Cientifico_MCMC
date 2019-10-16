#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import argparse
import numpy as np
from scipy import stats, integrate
import matplotlib.pyplot as plt

# Parser arguments
parser = argparse.ArgumentParser(
    description='Metropolis_Hastings generator.')
parser.add_argument('--sample_size', '--size',
                    type=int, default=100, metavar='N',
                    help='The smaple size of "random" numbers.')
parser.add_argument('--burn_in', '--burn',
                    type=int, default=0, metavar='N',
                    help='Number of samples to drop. (default: 0')
parser.add_argument('--sigma', '--sd',
                    type=float, default=.1, metavar='N',
                    help='Standard deviation of normal step (default: .1)')
parser.add_argument('--proposal', '--prop',
                    default='beta',
                    choices=['beta', 'normal', 'uniform'],
                    help='pick a step distribution (default: "beta")')
parser.add_argument('--log-interval', '--li',
                    type=int, default=100, metavar='N',
                    help='interval to print current status')
args = parser.parse_args()


def plot_random_walk(walk, rejected, n, p, r):
    # Plot random walk
    x = [w[0] for w in walk]
    y = [w[1] for w in walk]
    plt.plot(x, y, color='blue', label='walk', alpha=0.8)

    # Plot rejected proposals
    x = [r[0] for r in rejected]
    y = [r[1] for r in rejected]
    plt.scatter(x, y, marker='x', color='red', label='rejected')

    # Format plot
    plt.legend(loc='upper right')
    plt.title('M-H random walk with {} proposal.\n'
              '(n = {}, p = {:.2f}, r = {})'
              .format(args.proposal.title(), n, p, r))
    plt.savefig('walk_n={}_S={}_Prop={}.png'
                .format(n, args.sample_size, args.proposal),
                bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_sample(sample, posterior, n, p, r):
    # the histogram of the data
    m, bins, patches = plt.hist(sample,
                                'auto',
                                density=1,
                                facecolor='green',
                                alpha=0.5,
                                label='sample')

    # Plot density
    x_range = np.arange(0, 0.6, .001)
    y_density = [posterior(x) for x in x_range]
    norm_factor = integrate.simps(y_density, dx=.001)
    plt.plot(x_range, y_density / norm_factor, color='red',
             label='posterior density')

    # Format plot
    plt.legend(loc='upper right')
    plt.title('M-H sample using {} proposal.\n'
              '(n = {}, p = {:.2f}, r = {})'
              .format(args.proposal.title(), n, p, r))
    plt.savefig('hist_n={}_S={}_Prop={}.png'
                .format(n, args.sample_size, args.proposal),
                bbox_inches='tight', pad_inches=0)
    plt.show()


def acceptance_ratio(x_p, x_t, posterior, proposal):
    if x_p < 0 or x_p > 0.5:  # Out of support
        return 0
    elif args.proposal == 'beta':
        return min(1, posterior(x_p) / posterior(x_t) *
                   proposal(x_t) / proposal(x_p))
    elif args.proposal == 'uniform':
        return min(1, posterior(x_p) / posterior(x_t) *
                   proposal(x_t) / proposal(x_p))
    elif args.proposal == 'normal':
        return min(1, posterior(x_p) / posterior(x_t) *
                   proposal(x_t, x_p) / proposal(x_p, x_t))


def metropolis_hastings(sample_size, posterior, proposal, step):
    # Current step
    t = 0

    # Initial uniform sample
    x_t = np.random.uniform(0, 0.5)

    # Sample
    sample = []

    # Random walk
    walk = []
    rejected = []

    # Init Burn-in count
    burnt = 0

    # Sample until desired number
    while len(sample) < sample_size:
        # Save random walk
        walk.append((t, x_t))

        # Generate random step from proposed distribution
        if args.proposal == 'beta':
            x_p = step()
        elif args.proposal == 'uniform':
            x_p = step()
        elif args.proposal == 'normal':
            x_p = step(x_t)

        # Calculate the acceptance ratio
        alpha = acceptance_ratio(x_p, x_t, posterior, proposal)

        # Random uniform sample
        u = np.random.uniform()

        if u < alpha:  # Accept
            x_t = x_p

            if burnt < args.burn_in:   # Burn-in stage
                burnt += 1

                if burnt % args.log_interval == 0:
                    print('# Burnt:', burnt)

        else:  # Reject
            rejected.append((t, x_p))

        if burnt == args.burn_in:  # Sample stage
            sample.append(x_t)

            if len(sample) % args.log_interval == 0:
                print('# Samples:', len(sample))

        # Next step
        t += 1

    return (sample, walk, rejected)


def sample_from_posterior(n, p):
    # Simulated sum of n bernoulli random variables
    r = np.sum(stats.bernoulli.rvs(size=n, p=p))

    # Plot binomial trial
    print('n = {}, p = {}, r(success) = {}'.format(n, p, r))

    # Posterior to sample
    def posterior(p):
        if p >= 0 and p <= 0.5:
            return (p ** r) * ((1 - p)**(n - r)) * math.cos(np.pi * p)
        return 0.0

    if args.proposal == 'beta':
        # Beta distribution parameters
        alpha = r + 1
        beta = n - r + 1

        # Independent beta proposal
        def proposal(x):
            return stats.beta.pdf(x, alpha, beta)

        def step():
            return stats.beta.rvs(r + 1, n - r + 1)

    elif args.proposal == 'normal':
        # Normal dependent on past X_t
        def proposal(x, mu, sigma=args.sigma):
            return stats.norm.pdf(x, mu, sigma)

        def step(mu, sigma=args.sigma):
            return stats.norm.rvs(mu, sigma)

    elif args.proposal == 'uniform':
        # Independent uniform proposal
        def proposal(x):
            return stats.uniform.pdf(x, 0, 0.5)

        def step():
            return stats.uniform.rvs(0, 0.5)

    # Sample using Metropolis-Hastings
    (sample, walk, rejected) = metropolis_hastings(args.sample_size,
                                                   posterior,
                                                   proposal,
                                                   step)

    # Plot sample as random walk
    plot_random_walk(walk, rejected, n, p, r)

    # Plot sample
    plot_sample(sample, posterior, n, p, r)

    return sample


def main():
    # Print format to 3 decimal spaces and fix seed
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    # Sample from posterior with diffente parameters
    sample_from_posterior(n=5, p=1 / 3)
    sample_from_posterior(n=35, p=1 / 3)


if __name__ == "__main__":
    main()
