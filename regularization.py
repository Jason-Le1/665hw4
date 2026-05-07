#------------------------------------
# Author: T. D. Devlin 
#-----------------------------------

import math
from math import sin, pi
from random import random


def f(x):
    return sin(pi * x)


def generate_training_examples(n=2):
    xs = [random() * 2 - 1 for _ in range(n)]
    return [(x, f(x)) for x in xs]


def fit_without_reg(examples):
    """Computes values of w0 and w1 that minimize the sum-of-squared-errors cost function

    Args:
    - examples: a list of two (x, y) tuples, where x is the feature and y is the label
    """
    ## BEGIN YOUR CODE ##
    (x1, y1), (x2, y2) = examples

    w1 = (y2 - y1) / (x2 - x1)
    w0 = y1 - w1 * x1
    ## END YOUR CODE ##
    return w0, w1


def fit_with_reg(examples, lambda_hp, step_size, steps):
    """Computes values of w0 and w1 that minimize the regularized sum-of-squared-errors cost function

    Args:
    - examples: a list of two (x, y) tuples, where x is the feature and y is the label
    - lambda_hp: a float representing the value of the lambda hyperparameter; a larger value means more regularization
    - step_size: a float representing the learning rate of parameters between updates
    - steps: a float representing the number of gradient descent updates
    """
    (x1, y1), (x2, y2) = examples
    w0 = 0.0
    w1 = 0.0
    ## BEGIN YOUR CODE ##
    for _ in range(steps):
        e1 = y1 - (w0 + w1 * x1)
        e2 = y2 - (w0 + w1 * x2)

        g0 = -2 * (e1 + e2) + 2 * lambda_hp * w0
        g1 = -2 * (x1 * e1 + x2 * e2) + 2 * lambda_hp * w1

        w0 -= step_size * g0
        w1 -= step_size * g1
    ## END YOUR CODE ##
    return (w0, w1)


def test_error(w0, w1):
    n = 100
    xs = [i/n for i in range(-n, n + 1)]
    return sum((w0 + w1 * x - f(x)) ** 2 for x in xs) / len(xs)


if __name__ == "__main__": 
    
    ## BEGIN YOUR SIMULATION CODE ##
    trials = 1000
    step_size = 0.05
    lambda_hp = 1.0
    total_error_no_reg = 0.0
    total_error_reg = 0.0

    for _ in range(trials):
        # Sample two points
        examples = generate_training_examples(n=2)

        w0_no_reg, w1_no_reg = fit_without_reg(examples)
        w0_reg, w1_reg = fit_with_reg(examples, lambda_hp, step_size, trials)

        # Estimate the out-of-sample error of reg and no_reg
        total_error_no_reg += test_error(w0_no_reg, w1_no_reg)
        total_error_reg += test_error(w0_reg, w1_reg)

    avg_error_no_reg = total_error_no_reg / trials
    avg_error_reg = total_error_reg / trials

    print(f"Average test error without regularization: {avg_error_no_reg:.6f}")
    print(f"Average test error with regularization: {avg_error_reg:.6f}")
