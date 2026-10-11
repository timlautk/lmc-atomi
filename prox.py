# Copyright 2023 by Tim Tsz-Kit Lau
# License: MIT License

import numpy as np
from scipy.optimize import minimize_scalar
import pylops


def prox_conjugate(x, gamma, prox):
    return x - gamma * prox(x / gamma, 1/gamma)


def prox_square_loss(x, y, H, gamma):
    d = x.shape[0] * x.shape[1]
    return (pylops.Identity(d) + gamma * H.adjoint() * H).div(x + gamma * H.adjoint() * y)


def prox_laplace(x, gamma): 
    return np.sign(x) * np.maximum(np.abs(x) - gamma, 0)


def prox_uncentered_laplace(x, gamma, mu):
    return mu + prox_laplace(x - mu, gamma)


def prox_gaussian(x, gamma):
    return x / (2*gamma + 1)


def prox_gen_gaussian(x, gamma, p):
    if p == 4/3:
        xi = np.sqrt(x**2 + 256*gamma**3/729)
        prox = x + 4 * gamma / (3*2**(1/3)) * ((xi - x)**(1/3) - (xi + x)**(1/3))
    elif p == 3/2: 
        prox = x + 9 * gamma**2 * np.sign(x) * (1 - np.sqrt(1 + 16 * np.abs(x)/(9*gamma**2))) / 8
    elif p == 3:
        prox = np.sign(x) * (np.sqrt(1 + 12*gamma*np.abs(x)) - 1) / (6*gamma)
    elif p == 4:
        xi = np.sqrt(x**2 + 1/(27*gamma))
        prox = ((xi + x)/(8*gamma))**(1/3) - ((xi - x)/(8*gamma))**(1/3)
    return prox


def prox_huber(x, gamma, tau):    
    return x / (2*tau + 1) if np.abs(x) <= gamma * (2*tau + 1) / np.sqrt(2*tau) else x - gamma * np.sqrt(2 * tau) * np.sign(x)
    

def prox_max_ent(x, gamma, tau, kappa, p):
    return np.sign(x) * prox_gen_gaussian(1/(2*tau+1) * np.maximum(np.abs(x) - gamma, 0), kappa/(2*tau+1))


def prox_smoothed_laplace(x, gamma):
    return np.sign(x) * (gamma * np.abs(x) - gamma**2 - 1 + np.sqrt(np.abs(gamma*np.abs(x) - gamma**2 - 1)**2 + 4*gamma*np.abs(x))) / (2*gamma)


def prox_exp(x, gamma): 
    return x - gamma if x >= gamma else 0.


def prox_gamma(x, omega, kappa):
    return (x - omega + np.sqrt((x - omega)**2 + 4*kappa)) / 2


def prox_chi(x, kappa):
    return (x + np.sqrt(x**2 + 8*kappa)) / 4


def prox_uniform(x, omega):
    if x < -omega:
        p = -omega
    elif x > omega:
        p = omega
    else: 
        p = x
    return p


def prox_triangular(x, omega1, omega2):
    if x < 1 / omega1:
        p = (x + omega1 + np.sqrt((x - omega1)**2 + 4)) / 2
    elif x > 1 / omega2:
        p = (x + omega2 + np.sqrt((x - omega2)**2 + 4)) / 2
    else: 
        p = 0.
    return p


def prox_weibull(x, omega, kappa, p):
    f = lambda y: p * omega * y**p + y**2 - x * y - kappa
    res = minimize_scalar(f, bounds=(0, np.inf), method='bounded')
    return res.x


def prox_gen_inv_gaussian(x, omega, kappa, rho):
    f = lambda y: y**3 + (omega - x) * y**2 - kappa * y - rho
    res = minimize_scalar(f, bounds=(0, np.inf), method='bounded')
    return res.x


def prox_pearson_I(x, kappa1, kappa2, omega1, omega2):
    f = lambda y: y**3 - (omega1 + omega2 + x) * y**2 + (omega1*omega2 - kappa1 - kappa2 + (omega1 + omega2)*x)*y \
        - omega1*omega2*x + omega1*kappa2 + omega2*kappa1
    res = minimize_scalar(f, bounds=(omega1, omega2), method='bounded')
    return res.x

