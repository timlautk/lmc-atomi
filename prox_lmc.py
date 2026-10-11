# Copyright 2023 by Tim Tsz-Kit Lau
# License: MIT License

import os
from fastprogress import progress_bar
import fire

import numpy as np
from numpy.random import default_rng

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import scienceplots
plt.style.use(['science'])
plt.rcParams.update({
    "font.family": "serif",   # specify font family here
    "font.serif": ["Times"],  # specify font here
    } 
    )

from scipy.linalg import sqrtm
from scipy.stats import multivariate_normal

import prox


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

    def multivariate_gaussian(self, theta, mu, Sigma):
        Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
        N = np.sqrt((2*np.pi)**self.d * np.abs(Sigma_det))
        fac = np.einsum('...k,kl,...l->...', theta - mu, Sigma_inv, theta - mu)
        return np.exp(-fac / 2) / N

    def density_gaussian_mixture(self, theta): 
        den = [self.omegas[i] * self.multivariate_gaussian(theta, self.mus[i], self.Sigmas[i]) for i in range(len(self.mus))]
        return sum(den)

    def potential_gaussian_mixture(self, theta): 
        return -np.log(self.density_gaussian_mixture(theta))

    def multivariate_laplacian(self, theta):    
        return (self.alpha/2)**self.d * np.exp(-self.alpha * np.linalg.norm(theta - self.mu, ord=1, axis=-1))
    
    def prox_uncentered_laplace(self, theta, gamma, mu):
        return mu + prox.prox_laplace(theta - mu, gamma)
    
    def moreau_env_uncentered_laplace(self, theta):
        p = self.prox_uncentered_laplace(theta, self.lamda * self.alpha, self.mu)
        return self.alpha * np.linalg.norm(p - self.mu, ord=1, axis=-1) + np.linalg.norm(p - theta, ord=2, axis=-1)**2 / (2 * self.lamda)
    
    def smooth_multivariate_laplacian(self, theta):
        return (self.alpha/2)**self.d * np.exp(-self.moreau_env_uncentered_laplace(theta))
        # return np.exp(-self.moreau_env_uncentered_laplace(theta))

    def grad_density_multivariate_gaussian(self, theta, mu, Sigma):
        return self.multivariate_gaussian(theta, mu, Sigma) * np.linalg.inv(Sigma) @ (mu - theta)

    def grad_density_gaussian_mixture(self, theta):
        grad_den = [self.omegas[i] * self.grad_density_multivariate_gaussian(theta, self.mus[i], self.Sigmas[i]) for i in range(len(self.mus))]
        return sum(grad_den)

    def grad_potential_gaussian_mixture(self, theta):
        return -self.grad_density_gaussian_mixture(theta) / self.density_gaussian_mixture(theta)

    def hess_density_multivariate_gaussian(self, theta, mu, Sigma):
        Sigma_inv = np.linalg.inv(Sigma)
        return self.multivariate_gaussian(theta, mu, Sigma) * (Sigma_inv @ np.outer(theta - mu, theta - mu) @ Sigma_inv - Sigma_inv)

    def hess_density_gaussian_mixture(self, theta):
        hess_den = [self.omegas[i] * self.hess_density_multivariate_gaussian(theta, self.mus[i], self.Sigmas[i]) for i in range(len(self.mus))]
        return sum(hess_den)
        
    def hess_potential_gaussian_mixture(self, theta):
        density = self.density_gaussian_mixture(theta)
        grad_density = self.grad_density_gaussian_mixture(theta)
        hess_density = self.hess_density_gaussian_mixture(theta)
        return np.outer(grad_density, grad_density) / density**2 - hess_density / density

    def gd_update(self, theta, gamma): 
        return theta - gamma * self.grad_potential_gaussian_mixture(theta) 


    ## Proximal Gradient Langevin Dynamics (PGLD)
    def pgld(self, gamma):
        print("\nSampling with Proximal ULA:")
        rng = default_rng(self.seed)
        theta0 = rng.standard_normal(self.d)
        theta = []
        for _ in progress_bar(range(self.n)):        
            xi = rng.multivariate_normal(np.zeros(self.d), np.identity(self.d))
            theta0 = prox.prox_laplace(theta0, self.lamda * self.alpha)
            theta_new = self.gd_update(theta0, gamma) + np.sqrt(2*gamma) * xi
            theta.append(theta_new)    
            theta0 = theta_new
        return np.array(theta)
    

    ## Moreau--Yosida Unadjusted Langevin Algorithm (MYULA)
    def grad_Moreau_env(self, theta):
        return (theta - prox.prox_laplace(theta, self.lamda * self.alpha)) / self.lamda

    def prox_update(self, theta, gamma):
        return -gamma * self.grad_Moreau_env(theta)

    def myula(self, gamma):
        print("\nSampling with MYULA:")
        rng = default_rng(self.seed)
        theta0 = rng.standard_normal(self.d)
        theta = []
        for _ in progress_bar(range(self.n)):
            xi = rng.multivariate_normal(np.zeros(self.d), np.identity(self.d))
            theta_new = self.gd_update(theta0, gamma) + self.prox_update(theta0, gamma) + np.sqrt(2*gamma) * xi
            theta.append(theta_new)    
            theta0 = theta_new
        return np.array(theta)


    ## Moreau--Yosida regularized Metropolis-Adjusted Langevin Algorithm (MYMALA)
    def q_prob(self, theta1, theta2, gamma):
        return multivariate_normal(mean=self.gd_update(theta2, gamma) + self.prox_update(theta2, gamma), cov=2*gamma).pdf(theta1)


    def prob(self, theta_new, theta_old, gamma):
        density_ratio = ((self.density_gaussian_mixture(theta_new) * self.multivariate_laplacian(theta_new)) / 
                        (self.density_gaussian_mixture(theta_old) * self.multivariate_laplacian(theta_old)))
        q_ratio = self.q_prob(theta_old, theta_new, gamma) / self.q_prob(theta_new, theta_old, gamma)
        return density_ratio * q_ratio


    def mymala(self, gamma):
        print("\nSampling with MYMALA:")
        rng = default_rng(self.seed)
        theta0 = rng.standard_normal(self.d)
        theta = []
        for _ in progress_bar(range(self.n)):
            xi = rng.multivariate_normal(np.zeros(self.d), np.identity(self.d))
            theta_new = self.gd_update(theta0, gamma) + self.prox_update(theta0, gamma) + np.sqrt(2*gamma) * xi
            p = self.prob(theta_new, theta0, gamma)
            alpha = min(1, p)
            if rng.random() <= alpha:
                theta.append(theta_new) 
                theta0 = theta_new
        return np.array(theta), len(theta)


    ## Preconditioned Proximal ULA (PP-ULA)
    def preconditioned_gd_update(self, theta, gamma, M): 
        return theta - gamma * M @ self.grad_potential_gaussian_mixture(theta)

    def preconditioned_prox(self, x, gamma, Q, t=100): 
        rho = 1 / np.linalg.norm(Q, ord=2)
        eps = max(min(1, rho) - 1e-5, 1e-9)
        eta = rho - eps
        w = np.zeros_like(x)
        for _ in range(t):
            u = x - Q @ w
            w = w + eta * u - eta * prox.prox_laplace(w / eta + u, gamma / eta)
        return u

    def preconditioned_prox_update(self, theta, gamma, Q, t=100):
        return -gamma * np.linalg.inv(Q) @ (theta - self.preconditioned_prox(theta, self.lamda, Q, t)) / self.lamda

    def ppula(self, gamma, M, Q, t=100):
        print("\nSampling with PP-ULA:")
        rng = default_rng(self.seed)
        theta0 = rng.standard_normal(self.d)
        theta = []
        for _ in progress_bar(range(self.n)):
            xi = rng.multivariate_normal(np.zeros(self.d), np.identity(self.d))
            theta_new = self.preconditioned_gd_update(theta0, gamma, M) + self.preconditioned_prox_update(theta0, gamma, Q, t) + np.sqrt(2*gamma) * sqrtm(M) @ xi
            theta.append(theta_new)    
            theta0 = theta_new
        return np.array(theta)


    ## Forward-Backward Unadjusted Langevin Algorithm (FBULA)
    def grad_FB_env(self, theta):
        return (np.identity(theta.shape[0]) - self.lamda * self.hess_potential_gaussian_mixture(theta)) @ (theta - prox.prox_laplace(self.gd_update(theta, self.lamda), self.lamda * self.alpha)) / self.lamda

    def gd_FB_update(self, theta, gamma):
        return theta - gamma * self.grad_FB_env(theta)

    def fbula(self, gamma):
        print("\nSampling with FBULA:")
        rng = default_rng(self.seed)
        theta0 = rng.standard_normal(self.d)
        theta = []
        for _ in progress_bar(range(self.n)):
            xi = rng.multivariate_normal(np.zeros(self.d), np.identity(self.d))
            theta_new = self.gd_FB_update(theta0, gamma) + np.sqrt(2*gamma) * xi
            theta.append(theta_new)    
            theta0 = theta_new
        return np.array(theta)


    ## Bregman--Moreau Unadjusted Mirror-Langevin Algorithm (BMUMLA)
    def grad_mirror_hyp(self, theta, beta): 
        return np.arcsinh(theta / beta)

    def grad_conjugate_mirror_hyp(self, theta, beta):
        return beta * np.sinh(theta)

    def left_bregman_prox_ell_one_hypent(self, theta, beta, gamma):
        if isinstance(theta, float):
            if theta > beta * np.sinh(gamma):
                p = beta * np.sinh(np.arcsinh(theta / beta) - gamma)
            elif theta < beta * np.sinh(-gamma):
                p = beta * np.sinh(np.arcsinh(theta / beta) + gamma)
            else: 
                p = np.sqrt(theta ** 2 + beta ** 2) - beta
        else:
            p = np.array(len(theta))
            p1 = beta * np.sinh(np.arcsinh(theta / beta) - gamma)
            p2 = beta * np.sinh(np.arcsinh(theta / beta) + gamma)
            p3 = np.sqrt(theta ** 2 + beta ** 2) - beta
            p = np.where(theta > beta * np.sinh(gamma), p1, p3)
            p = np.where(theta < beta * np.sinh(-gamma), p2, p)
        return p

    def grad_BM_env(self, theta, beta):
        return 1/self.lamda * (theta**2 + beta**2)**(-.5) * (theta - self.left_bregman_prox_ell_one_hypent(theta, beta, self.lamda * self.alpha))

    def gd_BM_update(self, theta, gamma):
        return -gamma * self.grad_potential_gaussian_mixture(theta)

    def prox_BM_update(self, theta, beta, gamma):
        return -gamma * self.grad_BM_env(theta, beta)

    def lbmumla(self, gamma, beta, sigma):
        print("\nSampling with LBMUMLA: ")
        rng = default_rng(self.seed)
        theta0 = rng.standard_normal(self.d)
        theta = []
        for _ in progress_bar(range(self.n)):
            xi = rng.multivariate_normal(np.zeros(self.d), np.identity(self.d))
            theta_new = self.grad_mirror_hyp(theta0, beta) + self.gd_BM_update(theta0, gamma) + self.prox_BM_update(theta0, sigma, gamma) + np.sqrt(2*gamma) * (theta0**2 + beta**2)**(-.25) * xi
            theta_new = self.grad_conjugate_mirror_hyp(theta_new, beta)
            theta.append(theta_new) 
            theta0 = theta_new
        return np.array(theta)


## Main function
def prox_lmc_gaussian_mixture(gamma_pgld=5e-2, gamma_myula=5e-2, 
                                gamma_mymala=5e-2, gamma_ppula=5e-2, 
                                gamma_fbula=5e-2, gamma_lbmumla=5e-2,
                                lamda=0.01, alpha=.1, n=5, t=100, 
                                K=10000, seed=0):
    # Our 2-dimensional distribution will be over variables X and Y
    N = 300
    xmin, xmax = -5, 5
    ymin, ymax = -5, 5
    X = np.linspace(xmin, xmax, N)
    Y = np.linspace(ymin, ymax, N)
    X, Y = np.meshgrid(X, Y)


    # Mean vectors and covariance matrices
    mu1 = np.array([0., 0.])
    Sigma1 = np.array([[ 1. , -0.5], [-0.5,  1.]])
    mu2 = np.array([-2., 3.])
    Sigma2 = np.array([[0.5, 0.2], [0.2, 0.7]])
    mu3 = np.array([2., -3.])
    Sigma3 = np.array([[0.5, 0.1], [0.1, 0.9]])
    mu4 = np.array([3., 3.])
    Sigma4 = np.array([[0.8, 0.02], [0.02, 0.3]])
    mu5 = np.array([-2., -2.])
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


    # location parameter of Laplacian prior
    mu = np.zeros_like(mus[0])

    # weight vector
    omegas = np.ones(n) / n

    # The distribution on the variables X, Y packed into pos.
    pos = np.empty(X.shape + (2,))

    # Pack X and Y into a single 3-dimensional array
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    prox_lmc = ProximalLangevinMonteCarlo(mus, Sigmas, omegas, lamda, alpha, mu, K, seed)  
    
    Z = prox_lmc.density_gaussian_mixture(pos) * prox_lmc.multivariate_laplacian(pos)
    Z_smooth = prox_lmc.density_gaussian_mixture(pos) * prox_lmc.smooth_multivariate_laplacian(pos)


    ## Plot of the true Gaussian mixture with Laplacian prior
    print("\nPlotting the true Gaussian mixture with Laplacian prior...")
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

    # plt.show(block=False)
    # plt.pause(10)
    # plt.close()
    fig.savefig(f'./fig/fig_prox_n{n}_gamma{gamma_pgld}_lambda{lamda}_{K}_1.pdf', dpi=500)


    print("\nPlotting the Gaussian mixture with smoothed Laplacian prior...")
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')

    ax1.plot_surface(X, Y, Z_smooth, rstride=3, cstride=3, linewidth=1, antialiased=True, cmap=cm.viridis)
    ax1.view_init(45, -70)

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.contourf(X, Y, Z_smooth, zdir='z', offset=0, cmap=cm.viridis)
    ax2.view_init(90, 270)

    ax2.grid(False)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])

    # plt.show(block=False)
    # plt.pause(5)
    # plt.close()
    fig.savefig(f'./fig/fig_prox_n{n}_gamma{gamma_pgld}_lambda{lamda}_{K}_1_smooth.pdf', dpi=500)


    Z1 = prox_lmc.pgld(gamma_pgld)

    Z2 = prox_lmc.myula(gamma_myula)

    Z3, eff_K = prox_lmc.mymala(gamma_mymala)
    print(f'\nMYMALA percentage of effective samples: {eff_K / K}')

    M = np.array([[1.0, 0.1], [0.1, 0.5]])    
    Q = np.array([[1.0, 0.1], [0.1, 1.5]])
    Z4 = prox_lmc.ppula(gamma_ppula, M, Q, t)

    Z5 = prox_lmc.fbula(gamma_fbula)
    
    beta = np.array([0.7, 0.3])
    sigma = np.array([0.8, 0.2])
    Z6 = prox_lmc.lbmumla(gamma_lbmumla, beta, sigma)


    ## Plot of the true Gaussian mixture with 2d histograms of samples
    print("\nConstructing the 2D histograms of samples...")
    ran = [[xmin, xmax], [ymin, ymax]]
    fig3, axes = plt.subplots(2, 4, figsize=(17, 8))

    axes[0,0].contourf(X, Y, Z, cmap=cm.viridis)
    axes[0,0].set_title("True density", fontsize=16)

    axes[0,1].contourf(X, Y, Z_smooth, cmap=cm.viridis)
    axes[0,1].set_title("Smoothed density", fontsize=16)

    axes[0,2].hist2d(Z1[:,0], Z1[:,1], bins=100, range=ran, cmap=cm.viridis)
    axes[0,2].set_title("PGLD", fontsize=16)

    axes[0,3].hist2d(Z2[:,0], Z2[:,1], bins=100, range=ran, cmap=cm.viridis)
    axes[0,3].set_title("MYULA", fontsize=16)

    axes[1,0].hist2d(Z3[:,0], Z3[:,1], bins=100, range=ran, cmap=cm.viridis)
    axes[1,0].set_title("MYMALA", fontsize=16)

    axes[1,1].hist2d(Z4[:,0], Z4[:,1], bins=100, range=ran, cmap=cm.viridis)
    axes[1,1].set_title("PP-ULA", fontsize=16)

    axes[1,2].hist2d(Z5[:,0], Z5[:,1], bins=100, range=ran, cmap=cm.viridis)
    axes[1,2].set_title("FBULA", fontsize=16)

    axes[1,3].hist2d(Z6[:,0], Z6[:,1], bins=100, range=ran, cmap=cm.viridis)
    axes[1,3].set_title("LBMUMLA", fontsize=16)


    # plt.show(block=False)
    # plt.pause(5)
    # plt.close()
    fig3.savefig(f'./fig/fig_prox_n{n}_gamma{gamma_pgld}_lambda{lamda}_{K}_3.pdf', dpi=500)

    
    ## Plot of the true Gaussian mixture with KDE of samples
    print("\nConstructing the KDEs of samples...")
    fig2, axes = plt.subplots(2, 4, figsize=(17, 8))

    sns.set(font='serif', rc={'figure.figsize':(3.25, 3.5)})

    axes[0,0].contourf(X, Y, Z, cmap=cm.viridis)
    axes[0,0].set_title("True density", fontsize=16)

    axes[0,1].contourf(X, Y, Z_smooth, cmap=cm.viridis)
    axes[0,1].set_title("Smoothed density", fontsize=16)

    sns.kdeplot(x=Z1[:,0], y=Z1[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[0,2])
    axes[0,2].set_title("PGLD", fontsize=16)

    sns.kdeplot(x=Z2[:,0], y=Z2[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[0,3])
    axes[0,3].set_title("MYULA", fontsize=16)

    sns.kdeplot(x=Z3[:,0], y=Z3[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[1,0])
    axes[1,0].set_title("MYMALA", fontsize=16)

    sns.kdeplot(x=Z4[:,0], y=Z4[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[1,1])
    axes[1,1].set_title("PP-ULA", fontsize=16)

    sns.kdeplot(x=Z5[:,0], y=Z5[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[1,2])
    axes[1,2].set_title("FBULA", fontsize=16)

    sns.kdeplot(x=Z6[:,0], y=Z6[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[1,3])
    axes[1,3].set_title("LBMUMLA", fontsize=16)
    
    # plt.show(block=False)
    # plt.pause(5)
    # plt.close()
    fig2.savefig(f'./fig/fig_prox_n{n}_gamma{gamma_pgld}_lambda{lamda}_{K}_2.pdf', dpi=600)
    fig2.savefig(f'./fig/fig_prox_n{n}_gamma{gamma_pgld}_lambda{lamda}_{K}_2.eps', dpi=1200)

if __name__ == '__main__':
    if not os.path.exists('fig'):
        os.makedirs('fig')
    fire.Fire(prox_lmc_gaussian_mixture)