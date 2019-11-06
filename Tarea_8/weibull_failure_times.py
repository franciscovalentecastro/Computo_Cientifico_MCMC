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


def sample_from_weibull_posterior():
    # Simulate weibull failure times
    alpha = lmbd = 1
    c = b = 1
    n = 20

    # Failure times
    fail_times = stats.weibull_min.rvs(alpha, scale=1.0 / lmbd, size=n)
    print(fail_times)

    # Parameter priors
    def prior_alpha(x, log=False):
        if log:
            return stats.expon.logpdf(x, scale=1.0 / c)
        else:
            return stats.expon.pdf(x, scale=1.0 / c)

    def prior_lmbd(x, alpha_t, log=False):
        if log:
            return stats.gamma.logpdf(x, alpha_t, scale=1.0 / b)
        else:
            return stats.gamma.pdf(x, alpha_t, scale=1.0 / b)

    # Posterior to sample with metropolis hastings
    def posterior(x):
        alpha_t, lmbd_t = x
        likelihood = stats.weibull_min.pdf(fail_times, alpha_t,
                                           scale=1.0 / lmbd_t)

        return likelihood.prod() * \
            prior_lmbd(lmbd_t, alpha_t) * \
            prior_alpha(alpha_t)

    def log_posterior(x):
        alpha_t, lmbd_t = x
        log_likelihood = stats.weibull_min.logpdf(fail_times, alpha_t,
                                                  scale=1.0 / lmbd_t)

        return log_likelihood.sum() + \
            prior_lmbd(lmbd_t, alpha_t, log=True) + \
            prior_alpha(alpha_t, log=True)

    # Gamma on lmbd
    def proposal_1(x, x_prime):
        alpha_t, lmbd_t = x
        param1 = alpha_t * n
        param2 = b + (fail_times ** alpha_t).sum()

        return stats.gamma.logpdf(lmbd_t, param1, scale=1.0 / param2)

    def step_1(x):
        alpha_t, lmbd_t = x
        param1 = alpha_t * n
        param2 = b + (fail_times ** alpha_t).sum()

        return (alpha_t, stats.gamma.rvs(param1, scale=1.0 / param2))

    # Gamma on alpha
    def proposal_2(x, x_prime):
        alpha_t, lmbd_t = x
        param1 = n + 1
        param2 = - np.log(b) - np.log(fail_times.prod()) + c

        return stats.gamma.logpdf(lmbd_t, param1, scale=1.0 / param2)

    def step_2(x):
        alpha_t, lmbd_t = x
        param1 = n + 1
        param2 = - np.log(b) - np.log(fail_times.prod()) + c

        return (stats.gamma.rvs(param1, scale=1.0 / param2), lmbd_t)

    # Hierarchy on alpha and then lmbd
    def proposal_3(x, x_prime):
        alpha_p, lmbd_p = x_prime

        return stats.expon.logpdf(alpha_p, scale=1.0 / c) + \
            stats.gamma.logpdf(lmbd_p, alpha_p, scale=1.0 / b)

    def step_3(x):
        alpha_p = stats.expon.rvs(scale=1.0 / c)
        lmbd_p = stats.gamma.rvs(alpha_p, scale=1.0 / b)

        return (alpha_p, lmbd_p)

    # Random walk
    sgm = 1

    def proposal_4(x, x_prime):
        alpha_t, lmbd_t = x
        alpha_p, lmbd_p = x_prime

        return stats.norm.pdf(alpha_p, alpha_t, sgm)

    def step_4(x):
        alpha_t, lmbd_t = x

        return (stats.norm.rvs(alpha_t, sgm), lmbd_t)

    # Acceptance Ratio
    def log_acceptance_ratio(x_p, x_t, log_posterior, log_proposal):
        if x_p[0].all() < 0 or x_p[1] < 0:  # Out of support
            return -np.inf
        else:
            return min(0, log_posterior(x_p) + log_proposal(x_p, x_t) -
                       log_posterior(x_t) - log_proposal(x_t, x_p))

    # Sample using Metropolis-Hastings
    proposal = [proposal_1, proposal_2, proposal_3, proposal_4]
    step = [step_1, step_2, step_3, step_4]
    probs = [.25, .25, .25, .25]

    # Intial value for Metropolis-Hastings
    x_init = (np.random.uniform(0, 1), np.random.uniform(0, 1))

    (sample, walk, rejected) = \
        metropolis_hastings_hybrid_kernels(args.sample_size,
                                           x_init,
                                           log_posterior,
                                           proposal,
                                           log_acceptance_ratio,
                                           step,
                                           probs)

    # Plot sample
    plot_sample(sample, posterior)
    plot_walk(sample, rejected, posterior)

    return sample


def main():
    # Print format to 3 decimal spaces and fix seed
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    # Sample from posterior with different parameters
    sample_from_weibull_posterior()


if __name__ == "__main__":
    main()
