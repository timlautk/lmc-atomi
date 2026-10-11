# Copyright 2023 by Tim Tsz-Kit Lau
# MIT License

# To install JAX, see its documentations

import jax.numpy as jnp
from jax.scipy.linalg import sqrtm
from jax.scipy.optimize import minimize


def prox_laplace(x, gamma): 
    return jnp.sign(x) * jnp.maximum(jnp.abs(x) - gamma, 0)


def prox_gaussian(x, gamma):
    return x / (2*gamma + 1)


def prox_gen_gaussian(x, gamma, p):
    if p == 4/3:
        xi = jnp.sqrt(x**2 + 256*gamma**3/729)
        prox = x + 4 * gamma / (3*2**(1/3)) * ((xi - x)**(1/3) - (xi + x)**(1/3))
    elif p == 3/2: 
        prox = x + 9 * gamma**2 * jnp.sign(x) * (1 - jnp.sqrt(1 + 16 * jnp.abs(x)/(9*gamma**2))) / 8
    elif p == 3:
        prox = jnp.sign(x) * (jnp.sqrt(1 + 12*gamma*jnp.abs(x)) - 1) / (6*gamma)
    elif p == 4:
        xi = jnp.sqrt(x**2 + 1/(27*gamma))
        prox = ((xi + x)/(8*gamma))**(1/3) - ((xi - x)/(8*gamma))**(1/3)
    return prox


def prox_huber(x, gamma, tau):    
    return x / (2*tau + 1) if jnp.abs(x) <= gamma * (2*tau + 1) / jnp.sqrt(2*tau) else x - gamma * jnp.sqrt(2 * tau) * jnp.sign(x)
    

def prox_max_ent(x, gamma, tau, kappa, p):
    return jnp.sign(x) * prox_gen_gaussian(1/(2*tau+1) * jnp.maximum(jnp.abs(x) - gamma, 0), kappa/(2*tau+1))


def prox_smoothed_laplace(x, gamma):
    return jnp.sign(x) * (gamma * jnp.abs(x) - gamma**2 - 1 + jnp.sqrt(jnp.abs(gamma*jnp.abs(x) - gamma**2 - 1)**2 + 4*gamma*jnp.abs(x))) / (2*gamma)


def prox_exp(x, gamma): 
    return x - gamma if x >= gamma else 0.


def prox_gamma(x, omega, kappa):
    return (x - omega + jnp.sqrt((x - omega)**2 + 4*kappa)) / 2


def prox_chi(x, kappa):
    return (x + jnp.sqrt(x**2 + 8*kappa)) / 4


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
        p = (x + omega1 + jnp.sqrt((x - omega1)**2 + 4)) / 2
    elif x > 1 / omega2:
        p = (x + omega2 + jnp.sqrt((x - omega2)**2 + 4)) / 2
    else: 
        p = 0.
    return p


def prox_weibull(x, omega, kappa, p):
    f = lambda y: p * omega * y**p + y**2 - x * y - kappa
    res = minimize(f, bounds=(0,jnp.inf))
    return res.x


def prox_gen_inv_gaussian(x, omega, kappa, rho):
    f = lambda y: y**3 + (omega - x) * y**2 - kappa * y - rho
    res = minimize(f, bounds=(0, jnp.inf))
    return res.x


def prox_pearson_I(x, kappa1, kappa2, omega1, omega2):
    f = lambda y: y**3 - (omega1 + omega2 + x) * y**2 + (omega1*omega2 - kappa1 - kappa2 + (omega1 + omega2)*x)*y \
        - omega1*omega2*x + omega1*kappa2 + omega2*kappa1
    res = minimize(f, bounds=(omega1, omega2))
    return res.x
