#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import numpy as np
from scipy import stats

from metropolis_hastings_hybrid_kernels import *

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


def sample_from_posterior():
    # Parameters
    alpha = 1.8
    gamma = 0.01
    delta = 1.0

    # Nuclear plant failure times data
    t = [94.32, 15.72, 62.88, 125.76, 5.24, 31.44, 1.05, 1.05, 2.10, 10.48]
    p = [5, 1, 5, 14, 3, 19, 1, 1, 4, 22]

    print([p_i / t_i for t_i, p_i in zip(t, p)])

    # Posterior to sample with metropolis hastings
    def posterior(x):
        lmbd_t, beta_t = x
        prod = 1

        for idx, lmbd_t_i in enumerate(lmbd_t):
            A = lmbd_t_i ** (p[idx] + alpha - 1)
            B = np.exp(-(t[idx] + beta_t) * lmbd_t_i)

            prod *= A * B

        C = beta_t ** (10 * alpha + gamma - 1)
        D = np.exp(-delta * beta_t)

        return prod * C * D

    def log_posterior(x):
        lmbd_t, beta_t = x
        sm = 0

        for idx, lmbd_t_i in enumerate(lmbd_t):
            A = (p[idx] + alpha - 1) * np.log(lmbd_t_i)
            B = -(t[idx] + beta_t) * lmbd_t_i

            sm += A + B

        C = (10 * alpha + gamma - 1) * np.log(beta_t)
        D = -delta * beta_t

        return sm + C + D

    # Gamma on lmbd
    def proposal_lmbd_i(x, x_prime, index):
        lmbd_t, beta_t = x
        lmbd_p, beta_p = x_prime

        param1 = p[index] + alpha
        param2 = t[index] + beta_t

        return stats.gamma.logpdf(lmbd_p[index], param1, scale=1.0 / param2)

    def step_lmbd_i(x, index):
        lmbd_t, beta_t = x

        param1 = p[index] + alpha
        param2 = t[index] + beta_t

        lmbd_p = lmbd_t.copy()
        lmbd_p[index] = stats.gamma.rvs(param1, scale=1.0 / param2)

        return (lmbd_p, beta_t)

    # Gamma on beta
    def proposal_beta(x, x_prime):
        lmbd_t, beta_t = x
        lmbd_p, beta_p = x_prime

        param1 = 10 * alpha + gamma
        param2 = delta + lmbd_t.sum()

        return stats.gamma.logpdf(beta_p, param1, scale=1.0 / param2)

    def step_beta(x):
        lmbd_t, beta_t = x

        param1 = 10 * alpha + gamma
        param2 = delta + lmbd_t.sum()

        beta_p = stats.gamma.rvs(param1, scale=1.0 / param2)

        return (lmbd_t, beta_p)

    # Acceptance Ratio
    def log_acceptance_ratio(x_p, x_t, log_posterior, log_proposal):
        if x_p[0].any() <= 0 or x_p[1] <= 0:  # Out of support
            return -np.inf
        else:
            return min(0, log_posterior(x_p) + log_proposal(x_p, x_t) -
                       log_posterior(x_t) - log_proposal(x_t, x_p))

    # Sample using Metropolis-Hastings
    proposal = [lambda x, x_prime, index=idx:proposal_lmbd_i(x, x_prime, index)
                for idx in range(10)] + [proposal_beta]
    step = [lambda x, index=idx:step_lmbd_i(x, index)
            for idx in range(10)] + [step_beta]
    probs = [1.0 / 11.0] * 11

    # Intial value for Metropolis-Hastings
    x_init = (np.random.uniform(0, 1, 10), np.random.uniform(0, 1))

    # Metropolis-Hastings execution
    (sample, walk, rejected) = \
        metropolis_hastings_hybrid_kernels(args.sample_size,
                                           x_init,
                                           log_posterior,
                                           proposal,
                                           log_acceptance_ratio,
                                           step,
                                           probs)

    # Plot Results
    unpacked_walk = [tuple(list(elem[0]) + [elem[1]]) for elem in walk]
    unpacked_sample = [tuple(list(elem[0]) + [elem[1]]) for elem in sample]
    name = 'imgs/burn-in_nuclear_s={}_b={}.png' \
           .format(args.sample_size, args.burn_in)
    plot_individual_walk_mean(list(enumerate(unpacked_walk)), name)
    plot_individual_hist(list(enumerate(unpacked_sample)), '')

    return sample


def main():
    # Print format to 3 decimal spaces and fix seed
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    # Sample from posterior with different parameters
    sample_from_posterior()


if __name__ == "__main__":
    main()
