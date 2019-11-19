#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
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


def print_rejection_statistics(sample, count_rejected):
    # All rejections
    perc = 100 * count_rejected / len(sample)
    print('All : Sampled {} : Rejected {}: Rej. Percent {:.2f} %'
          .format(len(sample), count_rejected, perc))


def plot_individual_hist(sample, name, params=['alpha', 'beta']):
    # Number of parameters
    n = len(sample[0][1])

    # For each parameter
    for idx in range(n):
        # Hist sample
        smp = np.array([elem[1][idx] for elem in sample])

        plt.hist(smp, bins=20, alpha=.6, density=True)
        plt.title('Histogram of "{}" sample'.format(params[idx]))
        plt.savefig('{}_{}.png'.format(name, params[idx]),
                    bbox_inches='tight', pad_inches=0)
        plt.show()


def plot_individual_walk_mean(walk, burn_in, name,
                              params=['alpha', 'beta']):
    # Number of parameters
    n = len(walk[0][1])

    means = []

    # For each parameter
    for idx in range(n):
        # Plot walk
        X_wlk = [elem[0] for elem in walk]
        Y_wlk = np.array([elem[1][idx] for elem in walk])
        Y_mean = [np.mean(Y_wlk[:idx + 1]) for idx in range(len(Y_wlk))]

        plt.plot(X_wlk, Y_mean, '-', alpha=1.0,
                 label='param {}'.format(params[idx]))
        means.append(np.mean(Y_wlk))

        # Format plot
        plt.axvline(x=burn_in, color='r', label='Burn-in')
        plt.legend(loc='upper right')

    plt.savefig('{}.png'.format(name),
                bbox_inches='tight', pad_inches=0)
    plt.show()

    print('Converged to the following mean ')
    print(means)


def plot_individual_walk(walk, rejected, burn_in,
                         name, params=['alpha', 'beta']):
    # Number of parameters
    n = len(walk[0][1])

    # For each parameter
    for idx in range(n):
        # Plot walk
        X_wlk = [elem[0] for elem in walk]
        Y_wlk = [elem[1][idx] for elem in walk]

        plt.plot(X_wlk, Y_wlk, '-', alpha=1.0,
                 label='param {}'.format(params[idx]))

        # Plot rejected
        X_rej = [elem[0] for elem in rejected]
        Y_rej = [elem[1][idx] for elem in rejected]
        plt.plot(X_rej, Y_rej, 'x', alpha=.2, color='red',
                 label='rejected')

        # Format plot
        plt.axvline(x=burn_in, color='r', label='Burn-in')
        plt.legend(loc='best')

    plt.savefig('{}.png'.format(name),
                bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_walk(walk, rejected, posterior, name, params=['alpha', 'beta']):
    # Plot walk
    X_wlk = [elem[0] for elem in walk]
    Y_wlk = [elem[1] for elem in walk]
    plt.plot(X_wlk, Y_wlk, '-o', alpha=0.4, color='blue',
             label='accepted')

    # Plot rejected
    X_rej = [elem[0] for elem in rejected]
    Y_rej = [elem[1] for elem in rejected]
    plt.plot(X_rej, Y_rej, 'x', alpha=0.2, color='red',
             label='rejected')
    plt.legend(loc='upper right')

    # Get max and min of walk
    X_max = np.max(X_wlk + X_rej)
    X_min = np.min(X_wlk + X_rej)
    Y_max = np.max(Y_wlk + Y_rej)
    Y_min = np.min(Y_wlk + Y_rej)

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
    plt.xlabel(params[0])
    plt.ylabel(params[1])
    plt.savefig(name, bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_sample(sample, posterior, name, params=['alpha', 'beta']):
    # Plot sample
    X_smp = [elem[0] for elem in sample]
    Y_smp = [elem[1] for elem in sample]
    plt.plot(X_smp, Y_smp, 'o', alpha=0.4, color='blue',
             label='sample')
    plt.legend(loc='upper right')

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
    plt.xlabel(params[0])
    plt.ylabel(params[1])
    plt.savefig(name, bbox_inches='tight', pad_inches=0)
    plt.show()


# Acceptance Ratio
def log_acceptance_ratio(x_p, x_t, log_posterior, log_proposal):
    return min(0, log_posterior(x_p) + log_proposal(x_p, x_t) -
               log_posterior(x_t) - log_proposal(x_t, x_p))


def metropolis_hastings(sample_size,
                        x_initial,
                        log_posterior,
                        log_proposal,
                        step):
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

    # Counting statistics
    count_rejected = 0

    # Sample until desired number
    while len(sample) < sample_size:
        # Save random walk
        walk.append(x_t)

        # Generate random step from proposed distribution
        x_p = step(x_t)

        # Calculate the acceptance ratio
        rho = np.exp(log_acceptance_ratio(x_p, x_t,
                                          log_posterior,
                                          log_proposal))

        # Random uniform sample
        u = np.random.uniform()

        if u < rho:  # Accept
            x_t = x_p

        else:  # Reject
            rejected.append((t, x_p))
            if burnt == args.burn_in:
                count_rejected += 1

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

    # Rejection statistics
    print_rejection_statistics(sample, count_rejected)

    return (sample, walk, rejected)


def sample_from_problem_1(n):
    # Parametros de distribuciones
    alpha = 3
    beta = 100
    sigma_1 = args.sigma
    sigma_2 = args.sigma * 10

    # Simulate from gamma(alpha = 3, beta = 100)
    gamma_sample = stats.gamma.rvs(alpha, scale=1.0 / beta, size=n)
    r1 = gamma_sample.prod()
    r2 = gamma_sample.sum()

    # Posterior to sample with metropolis hastings
    def posterior(x):
        alpha_t, beta_t = x

        if alpha_t < 1.0 or alpha_t > 4.0 or beta_t <= 1.0:
            return 0

        quotient = (beta_t ** (n * alpha_t)) / (math.gamma(alpha_t) ** n)
        product = (r1 ** (alpha_t - 1.0)) * np.exp(-beta_t * (r2 + 1.0))

        return quotient * product

    def log_posterior(x):
        alpha_t, beta_t = x

        if alpha_t < 1.0 or alpha_t > 4.0 or beta_t <= 1.0:
            return -np.inf

        quotient = n * alpha_t * np.log(beta_t)
        quotient -= n * np.log(math.gamma(alpha_t))
        product = (alpha_t - 1.0) * np.log(r1) - beta_t * (r2 + 1)

        return quotient + product

    def proposal(x, x_prime):
        SIGMA = [[sigma_1 ** 2, 0],
                 [0, sigma_2 ** 2]]

        return stats.multivariate_normal.pdf(x_prime, x, SIGMA)

    def log_proposal(x, x_prime):
        SIGMA = [[sigma_1 ** 2, 0],
                 [0, sigma_2 ** 2]]

        return stats.multivariate_normal.logpdf(x_prime, x, SIGMA)

    def step(x):
        SIGMA = [[sigma_1 ** 2, 0],
                 [0, sigma_2 ** 2]]

        return stats.multivariate_normal.rvs(x, SIGMA)

    # Intial value for Metropolis-Hastings
    x_init = (np.random.uniform(1, 4), np.random.exponential(1))

    # Sample using Metropolis-Hastings
    (sample, walk, rejected) = metropolis_hastings(args.sample_size,
                                                   x_init,
                                                   log_posterior,
                                                   log_proposal,
                                                   step)

    # Plot sample
    name = 'imgs/sample_prob1_s={}_b={}_n={}.png'\
           .format(args.sample_size, args.burn_in, n)
    plot_sample(sample, posterior, name)
    name = 'imgs/walk_prob1_s={}_b={}_n={}.png'\
           .format(args.sample_size, args.burn_in, n)
    plot_walk(sample, [elem[1] for elem in rejected], posterior, name)
    name = 'imgs/burn-in_prob1_s={}_b={}_n={}'\
           .format(args.sample_size, args.burn_in, n)
    plot_individual_walk_mean(list(enumerate(walk)), args.burn_in, name)
    name = 'imgs/walk_prob1_s={}_b={}_n={}'\
           .format(args.sample_size, args.burn_in, n)
    plot_individual_walk(list(enumerate(walk)), rejected, args.burn_in, name)
    name = 'imgs/hist_prob1_s={}_b={}_n={}'\
           .format(args.sample_size, args.burn_in, n)
    plot_individual_hist(list(enumerate(sample)), name)

    return sample


def sample_from_problem_2(x_init):
    # Posterior to sample with metropolis hastings
    alpha = 3.7

    def posterior(x):
        return stats.gamma.pdf(x, alpha, scale=1)

    def log_posterior(x):
        return stats.gamma.logpdf(x, alpha, scale=1)

    # Normal proposal
    def proposal(x, x_prime):
        interger_part = math.modf(x)[1]
        if interger_part < 1.0:
            return 0

        return stats.gamma.pdf(x_prime, interger_part, scale=1)

    def log_proposal(x, x_prime):
        interger_part = math.modf(x)[1]
        if interger_part < 1.0:
            return -np.inf

        return stats.gamma.logpdf(x_prime, interger_part, scale=1)

    def step(x):
        interger_part = math.modf(x)[1]
        return stats.gamma.rvs(interger_part, scale=1)

    # Sample using Metropolis-Hastings
    (sample, walk, rejected) = metropolis_hastings(args.sample_size,
                                                   x_init,
                                                   log_posterior,
                                                   log_proposal,
                                                   step)

    sample = [(elem,) for elem in sample]
    walk = [(elem,) for elem in walk]
    rejected = [(elem[0], (elem[1],)) for elem in rejected]

    # Plot sample
    name = 'imgs/burn-in_prob2_s={}_b={}'\
           .format(args.sample_size, args.burn_in)
    plot_individual_walk_mean(list(enumerate(walk)),
                              args.burn_in, name)
    name = 'imgs/walk_prob2_s={}_b={}'\
           .format(args.sample_size, args.burn_in)
    plot_individual_walk(list(enumerate(walk)),
                         rejected, args.burn_in, name)
    name = 'imgs/hist_prob2_s={}_b={}'\
           .format(args.sample_size, args.burn_in)
    plot_individual_hist(list(enumerate(sample)), name)

    return sample


def sample_from_problem_3(x_init):
    # Parametros de distribuciones
    MU = [3, 5]
    SIGMA = [[1, 0.9], [0.9, 1]]

    # Posterior to sample with metropolis hastings
    def posterior(x):
        return stats.multivariate_normal.pdf(x, MU, SIGMA)

    def log_posterior(x):
        return stats.multivariate_normal.logpdf(x, MU, SIGMA)

    # Normal proposal
    sigma = args.sigma

    def proposal(x, x_prime):
        SIGMA = [[sigma ** 2, 0],
                 [0, sigma ** 2]]

        return stats.multivariate_normal.pdf(x_prime, x, SIGMA)

    def log_proposal(x, x_prime):
        SIGMA = [[sigma ** 2, 0],
                 [0, sigma ** 2]]

        return stats.multivariate_normal.logpdf(x_prime, x, SIGMA)

    def step(x):
        SIGMA = [[sigma ** 2, 0],
                 [0, sigma ** 2]]

        return stats.multivariate_normal.rvs(x, SIGMA)

    # Sample using Metropolis-Hastings
    (sample, walk, rejected) = metropolis_hastings(args.sample_size,
                                                   x_init,
                                                   log_posterior,
                                                   log_proposal,
                                                   step)

    # Plot sample
    name = 'imgs/sample_prob3_s={}_b={}_sd={}.png'\
           .format(args.sample_size, args.burn_in, args.sigma)
    plot_sample(sample, posterior, name, params=['mu_1', 'mu_2'])
    name = 'imgs/walk_prob3_s={}_b={}_sd={}.png'\
           .format(args.sample_size, args.burn_in, args.sigma)
    plot_walk(sample, [elem[1] for elem in rejected], posterior, name,
              params=['mu_1', 'mu_2'])
    name = 'imgs/burn-in_prob3_s={}_b={}_sd={}'\
           .format(args.sample_size, args.burn_in, args.sigma)
    plot_individual_walk_mean(list(enumerate(walk)), args.burn_in,
                              name, params=['mu_1', 'mu_2'])
    name = 'imgs/walk_prob3_s={}_b={}_sd={}'\
           .format(args.sample_size, args.burn_in, args.sigma)
    plot_individual_walk(list(enumerate(walk)), rejected,
                         args.burn_in, name,
                         params=['mu_1', 'mu_2'])
    name = 'imgs/hist_prob3_s={}_b={}_sd={}'\
           .format(args.sample_size, args.burn_in, args.sigma)
    plot_individual_hist(list(enumerate(sample)),
                         name, params=['mu_1', 'mu_2'])

    return sample


def main():
    # Print format to 3 decimal spaces and fix seed
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    # Excercise - 1
    # sample_from_problem_1(n=3)
    # sample_from_problem_1(n=30)

    # Excercise - 2
    # sample_from_problem_2(x_init=np.random.uniform(3, 4))
    # sample_from_problem_2(x_init=1000)

    # Excercise - 3
    # sample_from_problem_3(x_init=(np.random.uniform(2, 4),
                                  # np.random.uniform(4, 6)))
    # sample_from_problem_3(x_init=(1000, 1))


if __name__ == "__main__":
    main()
