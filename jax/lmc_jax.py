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

class LangevinMonteCarlo:
    def __init__(self, mus, Sigmas, omegas, nChains=4, K=1000, seed=0) -> None:
        super(LangevinMonteCarlo, self).__init__()
        self.mus = mus
        self.Sigmas = Sigmas
        self.omegas = omegas
        self.nChains = nChains
        self.n = K
        self.seed = seed
        self.d = mus[0].shape[0]

    def multivariate_gaussian(self, theta, mu, Sigma):
        return stats.multivariate_normal.pdf(theta, mu, Sigma)

    def density_gaussian_mixture(self, theta): 
        den = [self.omegas[i] * self.multivariate_gaussian(theta, self.mus[i], self.Sigmas[i]) for i in range(len(self.mus))]
        return sum(den)

    def potential_gaussian_mixture(self, theta): 
        return -jnp.log(self.density_gaussian_mixture(theta))

    @jax.jit
    def grad_density_multivariate_gaussian(self, theta, mu, Sigma):
        return self.multivariate_gaussian(theta, mu, Sigma) * jnp.linalg.inv(Sigma) @ (mu - theta)
    
    @jax.jit
    def grad_potential_gaussian_mixture(self, theta): 
        # return jax.grad(self.potential_gaussian_mixture)
        grad_den = [self.omegas[i] * self.grad_density_multivariate_gaussian(theta, self.mus[i], self.Sigmas[i]) for i in range(len(self.mus))]
        return sum(grad_den)

    @jax.jit
    def grad_density_gaussian_mixture(self, theta):
        grad_den = [self.omegas[i] * self.grad_density_multivariate_gaussian(theta, self.mus[i], self.Sigmas[i]) for i in range(len(self.mus))]
        return sum(grad_den)

    @jax.jit
    def grad_potential_gaussian_mixture(self, theta):
        return -self.grad_density_gaussian_mixture(theta) / self.density_gaussian_mixture(theta)

    @jax.jit
    def hess_density_multivariate_gaussian(self, theta, mu, Sigma):
        Sigma_inv = jnp.linalg.inv(Sigma)
        return self.multivariate_gaussian(theta, mu, Sigma) * (Sigma_inv @ jnp.outer(theta - mu, theta - mu) @ Sigma_inv - Sigma_inv)

    @jax.jit
    def hess_density_gaussian_mixture(self, theta):
        # return jax.hessian(potential_2d_gaussian_mixture)
        hess_den = [self.omegas[i] * self.hess_density_multivariate_gaussian(theta, self.mus[i], self.Sigmas[i]) for i in range(len(self.mus))]
        return sum(hess_den)
    
    @jax.jit
    def hess_potential_gaussian_mixture(self, theta):
        density = self.density_gaussian_mixture(theta)
        grad_density = self.grad_density_gaussian_mixture(theta)
        hess_density = self.hess_density_gaussian_mixture(theta)
        return jnp.outer(grad_density, grad_density) / density**2 - hess_density / density

    @jax.jit
    def gd_update(self, theta, gamma): 
        return theta - gamma * self.grad_potential_gaussian_mixture(theta)


    ## Unadjusted Langevin Algorithm (ULA)
    # Use jax.pmap to sample with multiple chains
    def ula(self, gamma):
        print("\nSampling with ULA:")
        key = jax.random.PRNGKey(self.seed)
        theta0 = jax.random.normal(key, (self.d,))
        theta = []
        for _ in progress_bar(range(self.n)):
            xi = jax.random.multivariate_normal(key, jnp.zeros(self.d), jnp.eye(self.d))
            theta_new = self.gd_update(theta0, gamma) + jnp.sqrt(2*gamma) * xi
            theta.append(theta_new)    
            theta0 = theta_new
        return jnp.array(theta)


    ## Metropolis-Adjusted Langevin Algorithm (MALA)
    def q_prob(self, theta1, theta2, gamma, mus, Sigmas, lambdas):
        return jnp.exp(-1/(4*gamma) * jnp.linalg.norm(theta1 - self.gd_update(theta2, mus, Sigmas, lambdas, gamma))**2)


    def prob(self, theta_new, theta_old, gamma, mus, Sigmas, lambdas):
        density_ratio = self.density_gaussian_mixture(theta_new, mus, Sigmas, lambdas) / self.density_gaussian_mixture(theta_old, mus, Sigmas, lambdas)
        q_ratio = self.q_prob(theta_old, theta_new, gamma, mus, Sigmas, lambdas) / self.q_prob(theta_new, theta_old, gamma, mus, Sigmas, lambdas)
        return density_ratio * q_ratio


    def mala(self, gamma):
        print("\nSampling with MALA:")
        key = jax.random.PRNGKey(self.seed)
        theta0 = jax.random.normal(key, (self.d,))
        theta = []
        for _ in progress_bar(range(self.n)):
            xi = jax.random.multivariate_normal(key, jnp.zeros(self.d), jnp.eye(self.d))
            theta_new = self.gd_update(theta0, gamma) + jnp.sqrt(2*gamma) * xi
            p = self.prob(theta_new, theta0, gamma)
            alpha = min(1, p)
            if random.random() <= alpha:
                theta.append(theta_new)    
                theta0 = theta_new
        return jnp.array(theta), len(theta)


    ## Preconditioned ULA 
    @jax.jit
    def preconditioned_gd_update(self, theta, gamma, M): 
        return theta - gamma * M @ self.grad_potential_gaussian_mixture(theta)

    # @jax.jit
    # def hess_preconditioned_gd_update(theta, mus, Sigmas, lambdas, gamma): 
    #     hess = hess_potential_2d_gaussian_mixture(theta, mus, Sigmas, lambdas)
    #     eig = np.linalg.eig(hess)
    #     M = hess + eig * np.eye(hess.shape[0])
    #     return theta - gamma * np.linalg.inv(M) @ grad_potential_2d_gaussian_mixture(theta, mus, Sigmas, lambdas)   


    def pula(self, gamma, M):
        print("\nSampling with Preconditioned Langevin Algorithm:")
        key = jax.random.PRNGKey(self.seed)
        theta0 = jax.random.normal(key, (self.d,))
        theta = []
        for _ in progress_bar(range(self.n)):
            xi = jax.random.multivariate_normal(key, jnp.zeros(self.d), jnp.eye(self.d))
            theta_new = self.preconditioned_gd_update(theta0, gamma, M) + jnp.sqrt(2*gamma) * sqrtm(M) @ xi
            theta.append(theta_new)    
            theta0 = theta_new
        return jnp.array(theta)


    # Preconditioning with inverse Hessian
    def ihpula(self, gamma):
        print("\nSampling with Inverse Hessian Preconditioned Unadjusted Langevin Algorithm:")
        key = jax.random.PRNGKey(self.seed)
        theta0 = jax.random.normal(key, (self.d,))
        theta = []    
        for _ in progress_bar(range(self.n)):
            xi = jax.random.multivariate_normal(key, jnp.zeros(self.d), jnp.eye(self.d))
            hess = self.hess_potential_gaussian_mixture(theta0)
            if len(self.mus) > 1:
                e = jnp.linalg.eigvals(hess)
                M = hess + (jnp.abs(min(e)) + 0.05) * jnp.eye(self.d)
                M = jnp.linalg.inv(M)
            else:
                M = jnp.linalg.inv(hess)
            theta_new = self.preconditioned_gd_update(theta0, gamma, M) + jnp.sqrt(2*gamma) * sqrtm(M) @ xi
            theta.append(theta_new)    
            theta0 = theta_new
        return jnp.array(theta)


    ## Mirror-Langevin Algorithm (MLA)
    def grad_mirror_hyp(self, theta, beta): 
        return jnp.arcsinh(theta / beta)

    def grad_conjugate_mirror_hyp(self, theta, beta):
        return beta * jnp.sinh(theta)

    def mla(self, gamma, beta):
        print("\nSampling with MLA: ")
        key = jax.random.PRNGKey(self.seed)
        theta0 = jax.random.normal(key, (self.d,))
        theta = []
        for _ in progress_bar(range(self.n)):
            xi = jax.random.multivariate_normal(key, jnp.zeros(self.d), jnp.eye(self.d))
            theta_new = self.grad_mirror_hyp(theta0, beta) - gamma * self.grad_potential_gaussian_mixture(theta0) + jnp.sqrt(2*gamma) * (theta0**2 + beta**2)**(-.25) * xi
            theta_new = self.grad_conjugate_mirror_hyp(theta_new, beta)
            theta.append(theta_new)    
            theta0 = theta_new
        return jnp.array(theta)



## Main function
def lmc_gaussian_mixture(gamma_ula=5e-2, gamma_mala=5e-2, 
                         gamma_pula=5e-2, gamma_ihpula=5e-2, 
                         gamma_mla=5e-2, nChains=4, n=5, K=5000, seed=0):
    # Our 2-dimensional distribution will be over variables X and Y
    N = 300
    X = jnp.linspace(-5, 5, N)
    Y = jnp.linspace(-5, 5, N)
    X, Y = jnp.meshgrid(X, Y)

    # Mean vectors and covariance matrices
    mu1 = jnp.array([0., 0.])
    Sigma1 = jnp.array([[ 1. , -0.5], [-0.5,  1.]])
    mu2 = jnp.array([-2., 3.])
    Sigma2 = jnp.array([[0.5, 0.2], [0.2, 0.7]])
    mu3 = jnp.array([2., -3.])
    Sigma3 = jnp.array([[0.5, 0.1], [0.1, 0.9]])
    mu4 = jnp.array([3., 3.])
    Sigma4 = jnp.array([[0.8, 0.02], [0.02, 0.3]])
    mu5 = jnp.array([-2., -2.])
    Sigma5 = jnp.array([[1.2, 0.05], [0.05, 0.8]])

    if n == 1:
        mus = [mu1]
        Sigmas = [Sigma1]
    elif n == 2: 
        mus = [mu1, mu2]
        Sigmas = [Sigma1, Sigma2]
    elif n == 3: 
        mus = [mu1, mu2, mu3]
        Sigmas = [Sigma1, Sigma2, Sigma3]
    elif n == 4: 
        mus = [mu2, mu3, mu4, mu5]
        Sigmas = [Sigma2, Sigma3, Sigma4, Sigma5]
    elif n == 5: 
        mus = [mu1, mu2, mu3, mu4, mu5]
        Sigmas = [Sigma1, Sigma2, Sigma3, Sigma4, Sigma5]

    # weight vector
    omegas = jnp.ones(n) / n

    # Pack X and Y into a single 3-dimensional array
    # pos = jnp.empty(X.shape + (2,))
    # pos = pos.at[:, :, 0].set(X)
    # pos = pos.at[:, :, 1].set(Y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    pos = jnp.array(pos)

    nChains = min(nChains, len(jax.devices()))

    lmc = LangevinMonteCarlo(mus, Sigmas, omegas, nChains, K, seed)

    # The distribution on the variables X, Y packed into pos.
    Z = lmc.density_gaussian_mixture(pos)

    ## Plot of the true Gaussian mixture
    print("\nPlotting the true Gaussian mixture...")
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')

    ax1.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True, cmap=cm.viridis)
    ax1.view_init(45, -70)

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.contourf(X, Y, Z, zdir='z', offset=0, cmap=cm.viridis)
    ax2.view_init(90, 270)

    ax2.grid(False)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])

    # plt.suptitle("True 2D Gaussian Mixture") 
    plt.show(block=False)
    plt.pause(5)
    plt.close()
    fig.savefig(f'./fig/fig_n{n}_gamma{gamma_ula}_{K}_jax_1.pdf', dpi=500)


    Z2 = lmc.ula(gamma_ula)

    Z3, eff_K = lmc.mala(gamma_mala)
    print(f'\nMALA percentage of effective samples: {eff_K / K}')

    M = jnp.array([[1.0, 0.1], [0.1, 0.5]])
    Z4 = lmc.pula(gamma_pula, M)

    Z5 = lmc.ihpula(gamma_ihpula)

    # beta = np.array([0.2, 0.8])
    beta = jnp.array([0.7, 0.3])
    Z6 = lmc.mla(gamma_mla, beta)


    ## Plot of the true Gaussian mixture with 2d histograms of samples
    print("\nConstructing the 2D histograms of samples...")
    fig3, axes = plt.subplots(2, 3, figsize=(13, 8))

    axes[0,0].contourf(X, Y, Z, cmap=cm.viridis)
    axes[0,0].set_title("True density", fontsize=16)

    axes[0,1].hist2d(Z2[:,0], Z2[:,1], bins=100, cmap=cm.viridis)
    axes[0,1].set_title("ULA", fontsize=16)

    axes[0,2].hist2d(Z3[:,0], Z3[:,1], bins=100, cmap=cm.viridis)
    axes[0,2].set_title("MALA", fontsize=16)

    axes[1,0].hist2d(Z4[:,0], Z4[:,1], bins=100, cmap=cm.viridis)
    axes[1,0].set_title("PULA", fontsize=16)

    axes[1,1].hist2d(Z5[:,0], Z5[:,1], bins=100, cmap=cm.viridis)
    axes[1,1].set_title("IHPULA", fontsize=16)

    axes[1,2].hist2d(Z6[:,0], Z6[:,1], bins=100, cmap=cm.viridis)
    axes[1,2].set_title("MLA", fontsize=16)

    plt.show(block=False)
    plt.pause(10)
    plt.close()
    fig3.savefig(f'./fig/fig_n{n}_gamma{gamma_ula}_{K}_jax_3.pdf', dpi=500)


    ## Plot of the true Gaussian mixture with KDE of samples
    print("\nConstructing the KDEs of samples...")
    fig2, axes = plt.subplots(2, 3, figsize=(13, 8))
    sns.set(rc={'figure.figsize':(3.25, 3.5)})

    axes[0,0].contourf(X, Y, Z, cmap=cm.viridis)
    axes[0,0].set_title("True density", fontsize=16)

    sns.kdeplot(x=Z2[:,0], y=Z2[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[0,1])
    axes[0,1].set_title("ULA", fontsize=16)

    sns.kdeplot(x=Z3[:,0], y=Z3[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[0,2])
    axes[0,2].set_title("MALA", fontsize=16)

    sns.kdeplot(x=Z4[:,0], y=Z4[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[1,0])
    axes[1,0].set_title("PULA", fontsize=16)

    sns.kdeplot(x=Z5[:,0], y=Z5[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[1,1])
    axes[1,1].set_title("IHPULA", fontsize=16)

    sns.kdeplot(x=Z6[:,0], y=Z6[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[1,2])
    axes[1,2].set_title("MLA", fontsize=16)

    plt.show(block=False)
    plt.pause(5)
    plt.close()
    fig2.savefig(f'./fig/fig_n{n}_gamma{gamma_ula}_{K}_jax_2.pdf', dpi=500)  




if __name__ == '__main__':
    if not os.path.exists('fig'):
        os.makedirs('fig')
    fire.Fire(lmc_gaussian_mixture)