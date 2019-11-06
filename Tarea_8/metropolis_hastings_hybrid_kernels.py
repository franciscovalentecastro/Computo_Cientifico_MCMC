#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Parser arguments
parser = argparse.ArgumentParser(
    description='Metropolis_Hastings generator.')
parser.add_argument('--sample_size', '--size',
                    type=int, default=100, metavar='N',
                    help='The sample size of "random" numbers.')
parser.add_argument('--burn_in', '--burn',
                    type=int, default=0, metavar='N',
                    help='Number of samples to drop. (default: 0')
parser.add_argument('--sigma', '--sd',
                    type=float, default=.1, metavar='N',
                    help='Standard deviation of normal step (default: .1)')
parser.add_argument('--log-interval', '--li',
                    type=int, default=100, metavar='N',
                    help='interval to print current status')
args = parser.parse_args()


def plot_walk(walk, rejected, posterior):
    # Plot walk
    X_wlk = [elem[0] for elem in walk]
    Y_wlk = [elem[1] for elem in walk]
    plt.plot(X_wlk, Y_wlk, '-o', alpha=.5, color='blue',
             label='posterior density')

    # Plot rejected
    X_rej = [elem[0] for elem in rejected]
    Y_rej = [elem[1] for elem in rejected]
    plt.plot(X_rej, Y_rej, 'x', alpha=.5, color='red',
             label='posterior density')

    # Get max and min of walk
    X_max = np.max(X_wlk)
    X_min = np.min(X_wlk)
    Y_max = np.max(Y_wlk)
    Y_min = np.min(Y_wlk)

    # Plot contour
    X_lin = np.linspace(X_min, X_max, 100)
    Y_lin = np.linspace(Y_max, Y_min, 100)

    # Create grid
    X, Y = np.meshgrid(X_lin, Y_lin, indexing='xy')
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    # Evaluate posterior
    Z = np.array([[posterior((cell[0], cell[1]))
                   for cell in row] for row in pos])

    # Plot contour map
    plt.contour(X, Y, Z, 20, cmap='RdGy')
    plt.show()


def plot_sample(sample, posterior):
    # Plot sample
    X_smp = [elem[0] for elem in sample]
    Y_smp = [elem[1] for elem in sample]
    plt.plot(X_smp, Y_smp, 'o', alpha=.5, color='blue',
             label='posterior density')

    # Get max and min of sample
    X_max = np.max(X_smp)
    X_min = np.min(X_smp)
    Y_max = np.max(Y_smp)
    Y_min = np.min(Y_smp)

    # Plot contour
    X_lin = np.linspace(X_min, X_max, 100)
    Y_lin = np.linspace(Y_max, Y_min, 100)

    # Create grid
    X, Y = np.meshgrid(X_lin, Y_lin, indexing='xy')
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    # Evaluate posterior
    Z = np.array([[posterior((cell[0], cell[1]))
                   for cell in row] for row in pos])

    # Plot contour map
    plt.contour(X, Y, Z, 20, cmap='RdGy')
    plt.show()


def metropolis_hastings_hybrid_kernels(sample_size,
                                       x_initial,
                                       log_posterior,
                                       log_proposal,
                                       log_acceptance_ratio,
                                       step,
                                       kernel_probs):
    # Current step
    t = 0

    # Initial uniform sample
    x_t = x_initial

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
        walk.append(x_t)

        # Select transition kernel
        kernel_idx = np.random.choice(len(log_proposal), p=kernel_probs)
        # print(kernel_idx)

        log_proposal_t = log_proposal[kernel_idx]
        step_t = step[kernel_idx]
        # print(log_proposal_t)
        # print(step_t)

        # Generate random step from proposed distribution
        x_p = step_t(x_t)
        # print(x_p)

        # Calculate the acceptance ratio
        alpha = np.exp(log_acceptance_ratio(x_p, x_t,
                                            log_posterior,
                                            log_proposal_t))
        # print(alpha)

        # Random uniform sample
        u = np.random.uniform()

        if u < alpha:  # Accept
            # print('Accepted')
            x_t = x_p

        else:  # Reject
            # print('Rejected')
            if burnt == args.burn_in:
                rejected.append(x_p)

        # print(u)
        # input()

        if burnt == args.burn_in:  # Sample stage
            sample.append(x_t)

            if len(sample) % args.log_interval == 0:
                print('# Samples:', len(sample))

        else:  # Burn-in stage
            burnt += 1

            if burnt % args.log_interval == 0:
                print('# Burnt:', burnt)

        # Next step
        t += 1

    return (sample, walk, rejected)


def sample_from_normal_posterior(rho):
    # Posterior distribution params
    mu_1 = mu_2 = 0
    sigma_1 = sigma_2 = 1

    # Construct vectors
    MU = [mu_1, mu_2]
    SIGMA = [[sigma_1 ** 2, rho * sigma_1 * sigma_2],
             [rho * sigma_1 * sigma_2, sigma_2 ** 2]]

    # Posterior to sample with metropolis hastings
    def posterior(x):
        return stats.multivariate_normal.pdf(x, MU, SIGMA)

    def log_posterior(x):
        return stats.multivariate_normal.logpdf(x, MU, SIGMA)

    # Hybrid kernel proposals

    # X_1 | X_2 normal
    def proposal_1(x, x_prime):
        mean = mu_1 + rho * (sigma_1 / sigma_2) * (x[1] - mu_2)
        sigma = (sigma_1 ** 2) * (1 - rho ** 2)

        return stats.norm.logpdf(x_prime[1], mean, sigma)

    def step_1(x):
        mean = mu_1 + rho * (sigma_1 / sigma_2) * (x[1] - mu_2)
        sigma = (sigma_1 ** 2) * (1 - rho ** 2)

        return (stats.norm.rvs(mean, sigma), x[1])

    # X_2 | X_1 normal
    def proposal_2(x, x_prime):
        mean = mu_2 + rho * (sigma_2 / sigma_1) * (x[0] - mu_1)
        sigma = (sigma_2 ** 2) * (1 - rho ** 2)

        return stats.norm.logpdf(x_prime[0], mean, sigma)

    def step_2(x):
        mean = mu_2 + rho * (sigma_2 / sigma_1) * (x[0] - mu_1)
        sigma = (sigma_2 ** 2) * (1 - rho ** 2)

        return (x[0], stats.norm.rvs(mean, sigma))

    # Acceptance Ratio
    def log_acceptance_ratio(x_p, x_t, log_posterior, log_proposal):
        return min(0, log_posterior(x_p) + log_proposal(x_p, x_t) -
                   log_posterior(x_t) - log_proposal(x_t, x_p))

    # Intial value for Metropolis-Hastings
    x_init = (np.random.uniform(-1, 1), np.random.uniform(-1, 1))

    # Sample using Metropolis-Hastings
    (sample, walk, rejected) = \
        metropolis_hastings_hybrid_kernels(args.sample_size,
                                           x_init,
                                           log_posterior,
                                           [proposal_1, proposal_2],
                                           log_acceptance_ratio,
                                           [step_1, step_2],
                                           [.5, .5])

    print(len(sample))
    print(len(rejected))
    print(len(rejected) / len(sample))

    # Plot sample
    plot_sample(sample, posterior)
    plot_walk(sample, rejected, posterior)

    return sample


def main():
    # Print format to 3 decimal spaces and fix seed
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    # Sample from posterior with different parameters
    sample_from_normal_posterior(rho=0.8)
    sample_from_normal_posterior(rho=0.99)


if __name__ == "__main__":
    main()
