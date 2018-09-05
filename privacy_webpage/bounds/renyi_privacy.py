#!/usr/bin/env python2.7

from scipy.optimize import minimize
import numpy as np


def renyi_delta_of_eps(eps, d_alpha):
    f = lambda alpha: alpha * d_alpha(alpha) - alpha * eps
    initial_value = 1 + 10**-20
    res = minimize(f, [initial_value], bounds = [(1 + 10**-20, None)])
    alpha = res.x[0]
    return np.exp(f(alpha))
