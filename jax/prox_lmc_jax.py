# Copyright 2023 by Tim Tsz-Kit Lau
# MIT License

# To install JAX, see its documentations
# Install libraries: pip install -U numpy matplotlib scipy seaborn fire fastprogress SciencePlots scikit-image pylops pyproximal jax blackjax optax

import os
from fastprogress import progress_bar
import fire

import random
import numpy as np
from numpy.random import default_rng
rng = default_rng()

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
import seaborn as sns
import scienceplots
plt.style.use(['science', 'grid'])
plt.rcParams.update({
    "font.family": "serif",   # specify font family here
    "font.serif": ["Times"],  # specify font here
    } 
    )


from scipy.linalg import sqrtm
from scipy.stats import kde
from scipy.integrate import quad, dblquad

import jax
from jax import grad, jit
import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy.stats as stats

import blackjax
import optax
from blackjax.types import PyTree
from optax._src.base import OptState

import proxop
import prox_jax

if jax.lib.xla_bridge.get_backend().platform == "cpu":
    from jax.scipy.linalg import sqrtm
else:
    def sqrtm(x):
        u, s, vh = jnp.linalg.svd(x)
        return (u * s**.5) @ vh


class ProximalLangevinMonteCarlo:
    def __init__(self, mus, Sigmas, omegas, lamda, alpha, mu, K=1000, seed=0) -> None:
        super(ProximalLangevinMonteCarlo, self).__init__()
        self.mus = mus
        self.Sigmas = Sigmas
        self.omegas = omegas
        self.lamda = lamda
        self.alpha = alpha
        self.mu = mu
        self.n = K
        self.seed = seed
        self.d = mus[0].shape[0]

        

def multivariate_gaussian(theta, mu, Sigma):
    """Return the multivariate Gaussian distribution on array theta."""
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * np.abs(Sigma_det))
    # This einsum call calculates (theta - mu)T.Sigma-1.(theta - mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', theta - mu, Sigma_inv, theta - mu)

    return np.exp(-fac / 2) / N


def density_2d_gaussian_mixture(theta, mus, Sigmas, lambdas): 
    K = len(mus)
    den = [lambdas[k] * multivariate_gaussian(theta, mus[k], Sigmas[k]) for k in range(K)]
    return sum(den)


def potential_2d_gaussian_mixture(theta, mus, Sigmas, lambdas): 
    return -np.log(density_2d_gaussian_mixture(theta, mus, Sigmas, lambdas))


def prior(theta, alpha):    
    n = theta.shape[0]
    return (alpha/2)**n * np.exp(-alpha * np.linalg.norm(theta, axis=-1))


def grad_density_multivariate_gaussian(theta, mu, Sigma):
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * np.abs(Sigma_det))
    fac = np.einsum('...k,kl,...l->...', theta - mu, Sigma_inv, theta - mu)    
    return -.5 * np.exp(-fac / 2) / N * Sigma_inv @ (theta - mu)


def grad_density_2d_gaussian_mixture(theta, mus, Sigmas, lambdas):
    K = len(mus)
    grad_den = [lambdas[k] * grad_density_multivariate_gaussian(theta, mus[k], Sigmas[k]) for k in range(K)]
    return sum(grad_den)


def grad_potential_2d_gaussian_mixture(theta, mus, Sigmas, lambdas):
    return - grad_density_2d_gaussian_mixture(theta, mus, Sigmas, lambdas) / density_2d_gaussian_mixture(theta, mus, Sigmas, lambdas)


def hess_density_multivariate_gaussian(theta, mu, Sigma):
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * np.abs(Sigma_det))
    fac = np.einsum('...k,kl,...l->...', theta - mu, Sigma_inv, theta - mu)
    return -.5 * np.exp(-fac / 2) / N * (Sigma_inv + Sigma_inv @ np.outer(theta - mu, theta - mu) @ Sigma_inv)


def hess_density_2d_gaussian_mixture(theta, mus, Sigmas, lambdas):
    K = len(mus)
    hess_den = [lambdas[k] * hess_density_multivariate_gaussian(theta, mus[k], Sigmas[k]) for k in range(K)]
    return sum(hess_den)
    

def hess_potential_2d_gaussian_mixture(theta, mus, Sigmas, lambdas):
    density = density_2d_gaussian_mixture(theta, mus, Sigmas, lambdas)
    grad_density = grad_density_2d_gaussian_mixture(theta, mus, Sigmas, lambdas)
    hess_density = hess_density_2d_gaussian_mixture(theta, mus, Sigmas, lambdas)
    return np.outer(grad_density, grad_density) / density**2 - hess_density / density


def gd_update(theta, mus, Sigmas, lambdas, gamma): 
    return theta - gamma * grad_potential_2d_gaussian_mixture(theta, mus, Sigmas, lambdas)


## Proximal Unadjusted Langevin Algorithm (PULA)
def prox_ula_gaussian_mixture(gamma, mus, Sigmas, lambdas, d=2, n=1000, seed=0):
    print("\nSampling with PULA... ")
    proxH = lambda x, T : ProxNest.operators.proximal_operators.l1_projection(x, T, delta, Psi=psi)
    np.random.seed(seed)
    theta0 = np.random.normal(0, 1, d)
    theta = []
    for _ in progress_bar(range(n)):
        xi = rng.multivariate_normal(np.zeros(d), np.eye(d))
        theta_new = gd_update(theta0, mus, Sigmas, lambdas, gamma) + np.sqrt(2*gamma) * xi
        theta.append(theta_new)    
        theta0 = theta_new
    return np.array(theta)


## Metropolis-Adjusted Langevin Algorithm (MALA)
def q_prob(theta1, theta2, gamma, mus, Sigmas, lambdas):
    return np.exp(-1/(4*gamma) * np.linalg.norm(theta1 - gd_update(theta2, mus, Sigmas, lambdas, gamma))**2)


def prob(theta_new, theta_old, gamma, mus, Sigmas, lambdas):
    density_ratio = density_2d_gaussian_mixture(theta_new, mus, Sigmas, lambdas) / density_2d_gaussian_mixture(theta_old, mus, Sigmas, lambdas)
    q_ratio = q_prob(theta_old, theta_new, gamma, mus, Sigmas, lambdas) / q_prob(theta_new, theta_old, gamma, mus, Sigmas, lambdas)
    return density_ratio * q_ratio


def prox_mala_gaussian_mixture(gamma, mus, Sigmas, lambdas, d=2, n=1000, seed=0):
    print("\nSampling with Proximal MALA... ")
    np.random.seed(seed)
    theta0 = np.random.normal(0, 1, d)
    theta = []
    for _ in progress_bar(range(n)):
        xi = rng.multivariate_normal(np.zeros(d), np.eye(d))
        theta_new = gd_update(theta0, mus, Sigmas, lambdas, gamma) + np.sqrt(2*gamma) * xi
        p = prob(theta_new, theta0, gamma, mus, Sigmas, lambdas)
        alpha = min(1, p)
        if random.random() <= alpha:
            theta.append(theta_new)    
            theta0 = theta_new
    return np.array(theta), len(theta)


## Preconditioned ULA 
def preconditioned_gd_update(theta, mus, Sigmas, lambdas, gamma, M): 
    return theta - gamma * M @ grad_potential_2d_gaussian_mixture(theta, mus, Sigmas, lambdas)


def prox_preconditioned_langevin_gaussian_mixture(gamma, mus, Sigmas, lambdas, M, d=2, n=1000, seed=0):
    print("\nSampling with Preconditioned Langevin Algorithm: ")
    np.random.seed(seed)
    theta0 = np.random.normal(0, 1, d)
    theta = []
    for _ in progress_bar(range(n)):
        xi = rng.multivariate_normal(np.zeros(d), np.eye(d))
        theta_new = preconditioned_gd_update(theta0, mus, Sigmas, lambdas, gamma, M) + np.sqrt(2*gamma) * sqrtm(M) @ xi
        theta.append(theta_new)    
        theta0 = theta_new
    return np.array(theta)


# Preconditioning with inverse Hessian
def prox_hess_preconditioned_langevin_gaussian_mixture(gamma, mus, Sigmas, lambdas, d=2, n=1000, seed=0):
    print("\nSampling with Inverse Hessian Preconditioned Unadjusted Langevin Algorithm: ")
    np.random.seed(seed)
    theta0 = np.random.normal(0, 1, d)
    theta = []    
    for _ in progress_bar(range(n)):
        xi = rng.multivariate_normal(np.zeros(d), np.eye(d))
        hess = hess_potential_2d_gaussian_mixture(theta0, mus, Sigmas, lambdas)
        # e, _ = np.linalg.eig(hess)
        # M = hess + np.abs(min(e)) * np.eye(hess.shape[0])
        # print(np.linalg.det(hess))
        M = np.linalg.inv(hess)
        theta_new = preconditioned_gd_update(theta0, mus, Sigmas, lambdas, gamma, M) + np.sqrt(2*gamma) * sqrtm(M) @ xi
        theta.append(theta_new)    
        theta0 = theta_new
    return np.array(theta)


## Mirror-Langevin Algorithm (MLA)
def grad_mirror_hyp(theta, beta): 
    return np.arcsinh(theta / beta)

def grad_conjugate_mirror_hyp(theta, beta):
    return beta * np.sinh(theta)

def prox_mla_gaussian_mixture(gamma, mus, Sigmas, lambdas, beta, d=2, n=1000, seed=0):
    print("\nSampling with Proximal MLA: ")
    np.random.seed(seed)
    theta0 = np.random.normal(0, 1, d)
    theta = []
    for _ in progress_bar(range(n)):
        xi = rng.multivariate_normal(np.zeros(d), np.eye(d))
        theta_new = grad_mirror_hyp(theta0, beta) - gamma * grad_potential_2d_gaussian_mixture(theta0, mus, Sigmas, lambdas) + np.sqrt(2*gamma) * (theta0**2 + beta**2)**(-.25) * xi
        theta_new = grad_conjugate_mirror_hyp(theta_new, beta)
        theta.append(theta_new)    
        theta0 = theta_new
    return np.array(theta)


# Cyclical step sizes
def cyclical_gd_update(theta, mus, Sigmas, lambdas, gamma): 
    return theta - gamma * grad_potential_2d_gaussian_mixture(theta, mus, Sigmas, lambdas)


def prox_cyclical_ula_gaussian_mixture(gamma, mus, Sigmas, lambdas, d=2, n=1000, seed=0):
    print("\nSampling with Cyclical ULA: ")
    np.random.seed(seed)
    theta0 = np.random.normal(0, 1, d)
    theta = []
    for _ in progress_bar(range(n)):
        xi = rng.multivariate_normal(np.zeros(d), np.eye(d))
        theta_new = cyclical_gd_update(theta0, mus, Sigmas, lambdas, gamma) + np.sqrt(2*gamma) * xi
        theta.append(theta_new)    
        theta0 = theta_new
    return np.array(theta)



# Contour algorithms




'''
def error(theta, mus, Sigmas, lambdas):
    density_2d_gaussian_mixture(theta, mus, Sigmas)
    np.sum()

    return 
'''


def plot_hist2d(z, title): 
    z0 = z[:,0]
    z1 = z[:,1]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    hist, xedges, yedges = np.histogram2d(z0, z1, bins=50, range=[[-5, 5], [-5, 5]], density=True)

    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 0.5 * np.ones_like(zpos)
    dz = hist.ravel()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
    ax.set_title(title)
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    plt.show()


def plot_contour_hist2d(z, title, bins=50):
    counts, xbins, ybins, image = plt.hist2d(z[:,0], z[:,1], bins=bins, norm=LogNorm(), cmap=cm.viridis)
    plt.colorbar()
    plt.title(title)
    plt.show()


## Main function
def langevin_gaussian_mixture(gamma_ula=7.5e-2, gamma_mala=7.5e-2, gamma_pula=8e-2, gamma_ihpula=5e-4, gamma_mla=5e-2, alpha=.1, n=2, K=5000):
    # Our 2-dimensional distribution will be over variables X and Y
    N = 100
    X = np.linspace(-8, 8, N)
    Y = np.linspace(-8, 8, N)
    X, Y = np.meshgrid(X, Y)


    # Mean vectors and covariance matrices
    mu1 = np.array([3., 3.])
    Sigma1 = np.array([[ 1. , -0.5], [-0.5,  1.]])
    mu2 = np.array([-4., 6.])
    Sigma2 = np.array([[0.5, 0.2], [0.2, 0.7]])
    mu3 = np.array([4., -6.])
    Sigma3 = np.array([[0.5, 0.1], [0.1, 0.9]])
    mu4 = np.array([5., 5.])
    Sigma4 = np.array([[0.8, 0.02], [0.02, 0.3]])
    mu5 = np.array([-4., -4.])
    Sigma5 = np.array([[1.2, 0.05], [0.05, 0.8]])


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
    lambdas = np.ones(n) / n

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    # The distribution on the variables X, Y packed into pos.
    
    Z = density_2d_gaussian_mixture(pos, mus, Sigmas, lambdas) * prior(pos, alpha)
    # Z = prior(pos, alpha)
    # Z = np.exp(np.log(density_2d_gaussian_mixture(pos, mus, Sigmas, lambdas)) - 0.005 * np.sum(np.abs(pos)))
    # print(Z.shape)

    ## Plot of the true Gaussian mixture
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')

    ax1.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True, cmap=cm.viridis)
    ax1.view_init(45, -70)
    # ax1.set_xticks([])
    # ax1.set_yticks([])
    # ax1.set_zticks([])
    # ax1.set_xlabel(r'$x_1$')
    # ax1.set_ylabel(r'$x_2$')


    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.contourf(X, Y, Z, zdir='z', offset=0, cmap=cm.viridis)
    ax2.view_init(90, 270)

    ax2.grid(False)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])
    # ax2.set_xlabel(r'$x_1$')
    # ax2.set_ylabel(r'$x_2$')

    # plt.suptitle("True 2D Gaussian Mixture") 
    plt.show()
    # plt.show(block=False)
    # plt.pause(10)
    # plt.close()
    # fig.savefig(f'./fig/fig_prox_{n}_1.pdf', dpi=500)


    Z2 = prox_ula_gaussian_mixture(gamma_ula, mus, Sigmas, lambdas, n=K)
    # Plot of samples from the Langevin algorithm
    # plot_hist2d(Z2, "Unadjusted Langevin Algorithm")
    # plot_contour_hist2d(Z2, "Unadjusted Langevin Algorithm (ULA)")

    sns.set(rc={'figure.figsize':(3.25, 3.5)})
    # sns.kdeplot(x=Z2[:,0], y=Z2[:,1], cmap=cm.viridis, fill=True, thresh=0, clip=(-5, 5)).set(title="Unadjusted Langevin Algorithm (ULA)")
    # plt.show()

    # sns.jointplot(x=Z2[:,0], y=Z2[:,1], kind='kde')
    # plt.show()

    # print(error(Z2))


    Z3, eff_K = prox_mala_gaussian_mixture(gamma_mala, mus, Sigmas, lambdas, n=K)
    print('MALA acceptance rate:', eff_K / K)
    # Plot of samples from the MALA algorithm
    # plot_hist2d(Z3, "Metropolis-Adjusted Langevin Algorithm (MALA)")
    # plot_contour_hist2d(Z3, "Metropolis-Adjusted Langevin Algorithm (MALA)")

    # sns.kdeplot(x=Z3[:,0], y=Z3[:,1], cmap=cm.viridis, fill=True, thresh=0, clip=(-5, 5)).set(title="Metropolis-Adjusted Langevin Algorithm (MALA)")
    # plt.show()

    # sns.jointplot(x=Z3[:,0], y=Z3[:,1], kind='kde')
    # plt.show()

    # print(error(Z3))


    M = np.array([[1.0, 0.1], [0.1, 0.5]])
    Z4 = prox_preconditioned_langevin_gaussian_mixture(gamma_pula, mus, Sigmas, lambdas, M, n=K)
    # Plot of samples from the preconditioned Langevin algorithm
    # plot_hist2d(Z4, "Preconditioned Unadjusted Langevin Algorithm (PULA)")
    # plot_contour_hist2d(Z4, "Preconditioned Unadjusted Langevin Algorithm (PULA)")

    # sns.kdeplot(x=Z4[:,0], y=Z4[:,1], cmap=cm.viridis, fill=True, thresh=0, clip=(-5, 5)).set(title="Preconditioned Unadjusted Langevin Algorithm (PULA)")
    # plt.show()

    # sns.jointplot(x=Z4[:,0], y=Z4[:,1], kind='kde')
    # plt.show()

    # print(error(Z4))


    Z5 = prox_hess_preconditioned_langevin_gaussian_mixture(gamma_ihpula, mus, Sigmas, lambdas, n=K)
    # Plot of samples from the preconditioned Langevin algorithm
    # plot_hist2d(Z5, "Inverse Hessian Preconditioned Unadjusted Langevin Algorithm")
    # plot_contour_hist2d(Z5, "Inverse Hessian Preconditioned Unadjusted Langevin Algorithm")

    # sns.kdeplot(x=Z5[:,0], y=Z5[:,1], cmap=cm.viridis, fill=True, thresh=0, clip=(-5, 5)).set(title="Inverse Hessian Preconditioned Unadjusted Langevin Algorithm")
    # plt.show()

    # sns.jointplot(x=Z5[:,0], y=Z5[:,1], kind='kde')
    # plt.show()

    # print(error(Z5))


    # beta = np.array([0.2, 0.8])
    beta = np.array([0.7, 0.3])
    Z6 = prox_mla_gaussian_mixture(gamma_mla, mus, Sigmas, lambdas, beta, n=K)
    # Plot of samples from the preconditioned Langevin algorithm
    # plot_hist2d(Z6, "Mirror-Langevin Algorithm (MLA)")
    # plot_contour_hist2d(Z6, "Mirror-Langevin Algorithm (MLA)")

    # sns.kdeplot(x=Z6[:,0], y=Z6[:,1], cmap=cm.viridis, fill=True, thresh=0, clip=(-5, 5)).set(title="Mirror-Langevin Algorithm (MLA)")
    # plt.show()

    # sns.jointplot(x=Z6[:,0], y=Z6[:,1], kind='kde')
    # plt.show()

    # print(error(Z6))


    ## Plot of the true Gaussian mixture with KDE of samples
    fig2, axes = plt.subplots(2, 3, figsize=(13, 8))
    # fig2.suptitle("True density and KDEs of samples") 

    axes[0,0].contourf(X, Y, Z, cmap=cm.viridis)
    axes[0,0].set_title("True density")

    sns.kdeplot(x=Z2[:,0], y=Z2[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[0,1])
    axes[0,1].set_title("ULA")

    sns.kdeplot(x=Z3[:,0], y=Z3[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[0,2])
    axes[0,2].set_title("MALA")

    sns.kdeplot(x=Z4[:,0], y=Z4[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[1,0])
    axes[1,0].set_title("PULA")

    sns.kdeplot(x=Z5[:,0], y=Z5[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[1,1])
    axes[1,1].set_title("IHPULA")

    sns.kdeplot(x=Z6[:,0], y=Z6[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[1,2])
    axes[1,2].set_title("MLA")

    plt.show()
    # plt.pause(5)
    # plt.close()
    # fig2.savefig(f'./fig/fig_prox_{n}_2.pdf', dpi=500)  




if __name__ == '__main__':
    if not os.path.exists('fig'):
        os.makedirs('fig')
    fire.Fire(langevin_gaussian_mixture)