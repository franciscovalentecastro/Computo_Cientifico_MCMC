#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import argparse
import pandas as pd
import numpy as np

# Parser arguments
parser = argparse.ArgumentParser(
    description='Bootstrap estimations.')
parser.add_argument('--resample', '--r',
                    type=int, default=10, metavar='N',
                    help='Number bootstrap samples to take. (default: 10')
args = parser.parse_args()


def bootstrap_sample(sample):
    # Sample size
    n = len(sample)

    # Sample n elements with replacement
    bsmpl = [random.choice(sample) for i in range(n)]

    return bsmpl


def bootstrap_ci(sample, resample, estimator, confidence=.95):
    # Estimate with sample
    est = estimator(sample)

    # Estimated values list
    bestimated_lst = []

    # Resample multiple times and estimate
    for idx in range(resample):
        bsmpl = bootstrap_sample(sample)
        bestimated_lst.append(estimator(bsmpl))

    # Estimate .95, .5 quantiles from sample
    alpha = 1.0 - confidence
    quantiles = np.quantile(bestimated_lst, [alpha / 2.0, 1.0 - alpha / 2.0])

    # Basic confidence interval
    basic_ci = (2 * est - quantiles[1], 2 * est - quantiles[0])

    # Percentile confidence interval
    percen_ci = (quantiles[0], quantiles[1])

    return (basic_ci, percen_ci)


def jacknife_bias_corrected(sample, estimator):
    # Estimate with sample
    est = estimator(sample)

    # Sample size
    n = len(sample)

    # Estimated values list
    jkestimated_lst = []

    # Jacknife estimator
    for idx in range(n):
        # Drop idx element
        jckknf_smpl = sample[:idx] + sample[idx + 1:]
        jkestimated_lst.append(estimator(jckknf_smpl))

    jkestimator = np.mean(jkestimated_lst)

    # Estimate bias
    bias = (n - 1.0) * (jkestimator - est)

    # Bias corrected
    bc_jkestimator = est - bias

    return bc_jkestimator


def main():
    # Print format to 3 decimal spaces and fix seed
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    # Excercise # 1

    # Gamma shape=3, scale=2. Sample from Excercise #1
    sample = [14.18, 10.99, 3.38, 6.76, 5.56, 1.26, 4.05, 4.61,
              1.78, 3.84, 4.69, 2.12, 2.39, 16.75, 4.19]
    print('Sample : {}'.format(sample), end='\n\n')
    print('Mean : {}'.format(np.mean(sample)), end='\n')

    # Calculate confidence interval
    ci = bootstrap_ci(sample, args.resample, np.mean)
    print('Basic CI : ({}, {})'.format(*ci[0]), end='\n')
    print('Percentile CI : ({}, {})'.format(*ci[1]), end='\n\n')

    # Excercise # 2

    # Read CD4 data from boot library in R
    cd4 = pd.read_csv('cd4.csv')
    x = cd4['baseline'].tolist()
    y = cd4['oneyear'].tolist()
    sample = [[tx, ty] for tx, ty in zip(x, y)]

    # Print sample
    print('Sample : {}'.format(sample), end='\n\n')

    # Define estimator
    def correlation_coef(sample):
        return np.corrcoef(sample, rowvar=False)[0, 1]
    print('Corr Coeff : ', correlation_coef(sample))

    # Calculate confidence interval
    ci = bootstrap_ci(sample, args.resample, correlation_coef)
    print('Basic CI : ({}, {})'.format(*ci[0]), end='\n')
    print('Percentile CI : ({}, {})'.format(*ci[1]), end='\n')

    # Calculate bias corrected estimator
    bias_corrected_estimator = \
        jacknife_bias_corrected(sample, correlation_coef)
    print('Bias Corrected Corr Coeff : {}'
          .format(bias_corrected_estimator), end='\n\n')


if __name__ == "__main__":
    main()
