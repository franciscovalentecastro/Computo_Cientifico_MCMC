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
                    type=float, default=.001, metavar='N',
                    help='Standard deviation of normal step (default: .001)')
parser.add_argument('--log-interval', '--li',
                    type=int, default=100, metavar='N',
                    help='interval to print current status')
args = parser.parse_args()


def sample_problem_ecology():
    # Data of presence of a species
    sample = [6, 4, 9, 7, 8, 2, 8, 7, 5, 5,
              3, 9, 4, 5, 9, 8, 7, 5, 3, 2]
    m = len(sample)

    # Prior distributions of N and p
    alpha = 1
    beta = 20

    N_max = 1000
    elements = list(range(0, N_max + 1))

    def prior_p(x):
        return stats.beta.pdf(x, alpha, beta)

    def log_prior_p(x):
        return stats.beta.logpdf(x, alpha, beta)

    def prior_N(x):
        #if x not in elements:
        #    return 0.0
        #else:
        return 1.0 / (N_max + 1.0)

    def log_prior_N(x):
        if x not in elements:
            return -np.inf
        else:
            return -np.log(N_max + 1.0)

    # Posterior to sample with metropolis hastings
    def posterior(x):
        N, p = x

        if p < 0.0 or p > 1.0 or N < 0 or N > N_max:
            return 0

        likelihood = stats.binom.pmf(sample, N, p).prod()

        return likelihood * prior_p(p) * prior_N(N)

    def log_posterior(x):
        N, p = x

        if p < 0.0 or p > 1.0 or N < 0 or N > N_max:
            return -np.inf

        log_likelihood = stats.binom.logpmf(sample, N, p).sum()

        return log_likelihood + log_prior_p(p) + log_prior_N(N)

    # Hybrid kernel proposals

    # Gibbs kernel proposals
    def proposal_gibbs_p(x, x_prime):
        N, p = x
        N_prime, p_prime = x_prime

        alpha_t = alpha + np.sum(sample)
        beta_t = beta + m * N - np.sum(sample)

        return stats.beta.logpdf(p_prime, alpha_t, beta_t)

    def step_gibbs_p(x):
        N, p = x

        alpha_t = alpha + np.sum(sample)
        beta_t = beta + m * N - np.sum(sample)

        return (N, stats.beta.rvs(alpha_t, beta_t))

    def proposal_prior_p(x, x_prime):
        N, p = x
        N_prime, p_prime = x_prime

        return stats.beta.logpdf(p_prime, alpha, beta)

    def step_prior_p(x):
        N, p = x

        return (N, stats.beta.rvs(alpha, beta))

    def proposal_norm_p(x, x_prime):
        N, p = x
        N_prime, p_prime = x_prime

        loc = p
        scale = args.sigma

        return stats.norm.logpdf(p_prime, loc, scale)

    def step_norm_p(x):
        N, p = x

        loc = p
        scale = args.sigma

        return (N, stats.norm.rvs(loc, scale))

    def proposal_unif_N(x, x_prime):
        N, p = x
        N_prime, p_prime = x_prime

        return stats.randint.logpmf(N_prime, 0, N_max + 1)

    def step_unif_N(x):
        N, p = x

        return (stats.randint.rvs(0, N_max + 1), p)

    def proposal_poisson(x, x_prime):
        N, p = x
        N_prime, p_prime = x_prime

        lmbd_t = N * (1 - p)

        return stats.poisson.logpmf(N_prime - np.max(sample), lmbd_t)

    def step_poisson(x):
        N, p = x

        lmbd_t = N * (1 - p)

        return (np.max(sample) + stats.poisson.rvs(lmbd_t), p)

    def proposal_rw(x, x_prime):
        N, p = x
        N_prime, p_prime = x_prime

        if N_prime - N == 1 or N_prime - N == -1:
            return np.log(.5)
        else:
            return -np.inf

    def step_rw(x):
        N, p = x

        return (N + np.random.choice([1, -1], p=[.5, .5]), p)

    def proposal_both_rw(x, x_prime):
        N, p = x
        N_prime, p_prime = x_prime

        if N_prime - N == 1 or N_prime - N == -1:
            return np.log(.5) + stats.norm.logpdf(p_prime, p, args.sigma)
        else:
            return -np.inf

    def step_both_rw(x):
        N, p = x

        return (N + np.random.choice([1, -1], p=[.5, .5]),
                stats.norm.rvs(p, args.sigma))

    # Sample using Metropolis-Hastings
    proposal = [proposal_gibbs_p, proposal_prior_p, proposal_norm_p,
                proposal_unif_N, proposal_poisson, proposal_rw,
                proposal_both_rw]
    step = [step_gibbs_p, step_prior_p, step_norm_p,
            step_unif_N, step_poisson, step_rw,
            step_both_rw]
    probs = [.3, .1, .1, .1, .1, .2, .1]

    # Intial value for Metropolis-Hastings
    x_init = (np.random.randint(np.max(sample), N_max + 1),
              np.random.uniform(0, 1))

    # Sample using Metropolis-Hastings
    (sample, walk, rejected) = \
        metropolis_hastings_hybrid_kernels(args.sample_size,
                                           x_init,
                                           log_posterior,
                                           proposal,
                                           step,
                                           probs)

    # Plot sample
    name = 'imgs/sample_normal_s={}_b={}.png'\
           .format(args.sample_size, args.burn_in)
    plot_sample(sample, posterior, name)
    name = 'imgs/walk_normal_s={}_b={}.png'\
           .format(args.sample_size, args.burn_in)
    plot_walk(sample, rejected, posterior, name)
    name = 'imgs/burn-in_normal_s={}_b={}.png'\
           .format(args.sample_size, args.burn_in)
    plot_individual_walk_mean(list(enumerate(walk)), name)
    plot_individual_hist(list(enumerate(sample)), '')

    return sample


def main():
    # Print format to 3 decimal spaces and fix seed
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    # Sample from posterior with different parameters
    sample_problem_ecology()


if __name__ == "__main__":
    main()
