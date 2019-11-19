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
                    type=float, default=.01, metavar='N',
                    help='Standard deviation of normal step (default: .01)')
parser.add_argument('--log-interval', '--li',
                    type=int, default=100, metavar='N',
                    help='interval to print current status')
parser.add_argument('--plot', '--p',
                    action='store_true',
                    help='plot dataset sample')
args = parser.parse_args()


def sample_problem_ecology():
    # Data of spending (Y, pesos) by age (X)
    X_smpl = [28, 17, 14, 51, 16, 59, 16, 54, 52, 16, 31, 31, 54, 26, 19, 13,
              59, 48, 54, 23, 50, 59, 55, 37, 61, 53, 56, 31, 34, 15, 41, 14,
              13, 13, 32, 46, 17, 52, 54, 25, 61, 15, 53, 39, 33, 52, 65, 35,
              65, 26, 54, 16, 47, 14, 42, 47, 48, 25, 15, 46, 31, 50, 42, 23,
              17, 47, 32, 65, 45, 28, 12, 22, 30, 36, 33, 16, 39, 50, 13, 23,
              50, 34, 19, 46, 43, 56, 52, 42, 48, 55, 37, 21, 45, 64, 53, 16,
              62, 16, 25, 62]

    Y_smpl = [493, 165, 9, 0, 72, 0, 89, 0, 0, 70, 79, 96, 0, 1127, 548, 4,
              0, 0, 0, 1522, 0, 0, 0, 0, 0, 0, 0, 80, 5, 38, 0, 11, 8, 4, 31,
              0, 174, 0, 0, 1305, 0, 39, 0, 0, 18, 0, 0, 4, 0, 1102, 0, 94, 0,
              13, 0, 0, 0, 1308, 33, 0, 90, 0, 0, 1466, 156, 0, 39, 0, 0, 496,
              2, 1368, 190, 0, 12, 76, 0, 0, 5, 1497, 0, 6, 533, 0, 0, 0, 0, 0,
              0, 0, 0, 1090, 0, 0, 0, 93, 0, 88, 1275, 0]
    X_smpl, Y_smpl = zip(*sorted(zip(X_smpl, Y_smpl)))
    X_smpl = np.array(X_smpl)
    Y_smpl = np.array(Y_smpl)
    n = len(X_smpl)

    if args.plot:
        # Plot Y and X
        plt.plot(X_smpl, Y_smpl)
        plt.scatter(X_smpl, Y_smpl)
        plt.show()

    # Prior distributions of a, b and c
    mu_a = 35
    sd_a = 5
    alpha_b = 3
    beta_b = 3 / 950
    alpha_c = 2
    beta_c = 2 / 5

    def prior_a(x):
        return stats.norm.pdf(x, mu_a, sd_a)

    def log_prior_a(x):
        return stats.norm.logpdf(x, mu_a, sd_a)

    def prior_b(x):
        return stats.gamma.pdf(x, alpha_b, scale=1 / beta_b)

    def log_prior_b(x):
        return stats.gamma.logpdf(x, alpha_b, scale=1 / beta_b)

    def prior_c(x):
        return stats.gamma.pdf(x, alpha_c, scale=1 / beta_c)

    def log_prior_c(x):
        return stats.gamma.logpdf(x, alpha_c, scale=1 / beta_c)

    # Posterior to sample with metropolis hastings
    def link(x, a, b, c):
        return c * np.exp(-((x - a) ** 2) / (2 * (b ** 2)))

    def posterior(x):
        a, b, c = x

        likelihood = [0.0] * n
        a_np = np.array([a] * n)
        b_np = np.array([b] * n)
        c_np = np.array([c] * n)

        lmbd = link(X_smpl, a_np, b_np, c_np)
        likelihood = stats.poisson.pmf(Y_smpl, lmbd)
        likelihood = np.prod(likelihood)

        return likelihood * prior_a(a) * prior_b(b) * prior_c(c)

    def log_posterior(x):
        a, b, c = x

        log_likelihood = [0.0] * n
        a_np = np.array([a] * n)
        b_np = np.array([b] * n)
        c_np = np.array([c] * n)

        lmbd = link(X_smpl, a_np, b_np, c_np)
        log_likelihood = stats.poisson.logpmf(Y_smpl, lmbd)
        log_likelihood = np.sum(log_likelihood)

        return (log_likelihood + log_prior_a(a) +
                log_prior_b(b) + log_prior_c(c))

    # Hybrid kernel proposals

    # RWMH proposals
    def proposal_rw_a(x, x_prime):
        a, b, c = x
        a_prime, b_prime, c_prime = x_prime

        return stats.norm.logpdf(a_prime, a, args.sigma)

    def step_rw_a(x):
        a, b, c = x

        return (stats.norm.rvs(a, args.sigma), b, c)

    def proposal_rw_b(x, x_prime):
        a, b, c = x
        a_prime, b_prime, c_prime = x_prime

        return stats.norm.logpdf(b_prime, b, args.sigma)

    def step_rw_b(x):
        a, b, c = x

        return (a, stats.norm.rvs(b, args.sigma), c)

    def proposal_rw_c(x, x_prime):
        a, b, c = x
        a_prime, b_prime, c_prime = x_prime

        return stats.norm.logpdf(c_prime, c, args.sigma * 1000)

    def step_rw_c(x):
        a, b, c = x

        return (a, b, stats.norm.rvs(c, args.sigma * 1000))

    # Sample using Metropolis-Hastings
    proposal = [proposal_rw_a, proposal_rw_b, proposal_rw_c]
    step = [step_rw_a, step_rw_b, step_rw_c]
    probs = [.3, .3, .4]

    # Intial value for Metropolis-Hastings
    x_init = (np.random.uniform(20, 30),
              np.random.uniform(0, 10),
              np.random.uniform(1300, 1500))

    # Sample using Metropolis-Hastings
    (sample, walk, rejected) = \
        metropolis_hastings_hybrid_kernels(args.sample_size,
                                           x_init,
                                           log_posterior,
                                           proposal,
                                           step,
                                           probs)

    # Plot sample ab
    sample_ab = [(elem[0], elem[1]) for elem in sample]
    name = 'imgs/sample_market_ab_s={}_b={}.png'\
           .format(args.sample_size, args.burn_in)
    plot_sample(sample_ab, lambda x: posterior((x[0], x[1], 1400)),
                name, params=['a', 'b'])
    # Plot sample bc
    sample_bc = [(elem[1], elem[2]) for elem in sample]
    name = 'imgs/sample_market_bc_s={}_b={}.png'\
           .format(args.sample_size, args.burn_in)
    plot_sample(sample_bc, lambda x: posterior((24, x[0], x[1])),
                name, params=['b', 'c'])
    # Plot sample ac
    sample_ac = [(elem[0], elem[2]) for elem in sample]
    name = 'imgs/sample_market_ac_s={}_b={}.png'\
           .format(args.sample_size, args.burn_in)
    plot_sample(sample_ac, lambda x: posterior((x[0], 4, x[1])),
                name, params=['a', 'c'])
    # Plot burn-in
    name = 'imgs/burn-in_market_s={}_b={}.png'\
           .format(args.sample_size, args.burn_in)
    plot_individual_walk_mean(list(enumerate(walk)),
                              args.burn_in, name)
    # Plot walk
    name = 'imgs/walk_market_s={}_b={}'\
           .format(args.sample_size, args.burn_in)
    plot_individual_walk(list(enumerate(walk)), rejected,
                         args.burn_in, name)
    # Plot hist
    name = 'imgs/hist_market_s={}_b={}'\
           .format(args.sample_size, args.burn_in)
    plot_individual_hist(list(enumerate(sample)), name)

    return sample


def main():
    # Print format to 3 decimal spaces and fix seed
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    # Sample from posterior with different parameters
    sample_problem_ecology()


if __name__ == "__main__":
    main()
