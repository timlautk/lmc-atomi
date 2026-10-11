# Copyright 2023 by Tim Tsz-Kit Lau
# MIT License

# To install JAX, see its documentations
# Install libraries: pip install -U numpy matplotlib scipy seaborn fire fastprogress SciencePlots scikit-image pylops pyproximal jax blackjax optax

import os
import itertools
from fastprogress import progress_bar
from typing import NamedTuple
import multiprocessing
import fire

import numpy as np
import random

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count())
import jax
from jax import grad, jit
import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy.stats as stats

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import scienceplots
plt.style.use(['science', 'grid'])
plt.rcParams.update({
    "font.family": "serif",   # specify font family here
    "font.serif": ["Times"],  # specify font here
    } 
    )

if jax.lib.xla_bridge.get_backend().platform == "cpu":
    from jax.scipy.linalg import sqrtm
else:
    def sqrtm(x):
        u, s, vh = jnp.linalg.svd(x)
        return (u * s**.5) @ vh
    
import prox_jax


class LangevinMonteCarloLaplacian:
    def __init__(self, mus, alphas, omegas, lamda, K=1000, seed=0) -> None:
        super(LangevinMonteCarloLaplacian, self).__init__()
        self.mus = mus
        self.alphas = alphas
        self.omegas = omegas
        self.lamda = lamda        
        self.n = K
        self.seed = seed
        self.d = mus[0].shape[0]  

    

