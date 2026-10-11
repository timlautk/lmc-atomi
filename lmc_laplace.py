# Copyright 2023 by Tim Tsz-Kit Lau
# License: MIT License

import os
from fastprogress import progress_bar
import fire
import time

import numpy as np
from numpy.random import default_rng
from scipy.linalg import sqrtm
from scipy.stats import multivariate_normal
from multivariate_laplace import multivariate_laplace
import ot
import ot.plot

import matplotlib as mpl
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

import prox

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
        self.Sigmas = [np.eye(self.d) * 2 / self.alphas[i] for i in range(len(self.alphas))]

    def multivariate_laplacian(self, theta, mu, alpha):
        return (alpha/2)**self.d * np.exp(-alpha * np.linalg.norm(theta - mu, ord=1, axis=-1))
    
    def density_laplacian_mixture(self, theta): 
        den = [self.omegas[i] * self.multivariate_laplacian(theta, self.mus[i], self.alphas[i]) for i in range(len(self.alphas))]
        return sum(den)
    
    def potential_laplacian_mixture(self, theta): 
        return -np.log(self.density_laplacian_mixture(theta))
    
    def prox_uncentered_laplace(self, theta, gamma, mu):
        return mu + prox.prox_laplace(theta - mu, gamma)
    
    def moreau_env_uncentered_laplace(self, theta, mu, alpha):
        prox = self.prox_uncentered_laplace(theta, self.lamda * alpha, mu)
        return alpha * np.linalg.norm(prox - mu, ord=1, axis=-1) + np.linalg.norm(prox - theta, ord=2, axis=-1)**2 / (2 * self.lamda)
    
    def smooth_multivariate_laplacian(self, theta, mu, alpha):
        return (alpha/2)**self.d * np.exp(-self.moreau_env_uncentered_laplace(theta, mu, alpha))
    
    def smooth_density_laplacian_mixture(self, theta): 
        den = [self.omegas[i] * self.smooth_multivariate_laplacian(theta, self.mus[i], self.alphas[i]) for i in range(len(self.alphas))]
        return sum(den)
    
    def smooth_potential_laplacian_mixture(self, theta): 
        return -np.log(self.smooth_density_laplacian_mixture(theta))

    def grad_smooth_density_multivariate_laplacian(self, theta, mu, alpha):        
        return self.smooth_multivariate_laplacian(theta, mu, alpha) * (self.prox_uncentered_laplace(theta, self.lamda * alpha, mu) - theta) / self.lamda
    
    def grad_smooth_density_laplacian_mixture(self, theta):
        grad_den = [self.omegas[i] * self.grad_smooth_density_multivariate_laplacian(theta, self.mus[i], self.alphas[i]) for i in range(len(self.alphas))]
        return sum(grad_den)
    
    def grad_smooth_potential_laplacian_mixture(self, theta):
        return -self.grad_smooth_density_laplacian_mixture(theta) / self.smooth_density_laplacian_mixture(theta)
    
    def hess_smooth_density_multivariate_laplacian(self, theta, mu, alpha):
        grad_laplace = (theta - self.prox_uncentered_laplace(theta, self.lamda * alpha, mu)) / self.lamda
        return (alpha/2)**self.d * np.exp(-alpha * np.linalg.norm(theta, ord=1, axis=-1)) * ( - np.eye(self.d) + np.outer(grad_laplace, grad_laplace))

    def hess_smooth_density_laplacian_mixture(self, theta):
        hess_den = [self.omegas[i] * self.hess_smooth_density_multivariate_laplacian(theta, self.mus[i], self.alphas[i]) for i in range(len(self.alphas))]
        return sum(hess_den)
        
    def hess_smooth_potential_laplacian_mixture(self, theta):
        density = self.smooth_density_laplacian_mixture(theta)
        grad_density = self.grad_smooth_density_laplacian_mixture(theta)
        hess_density = self.hess_smooth_density_laplacian_mixture(theta)
        return np.outer(grad_density, grad_density) / density**2 - hess_density / density
    
    def gd_update(self, theta, gamma): 
        return theta - gamma * self.grad_smooth_potential_laplacian_mixture(theta) 


    ## True samples
    def true_samples(self):
        print("\nGenerating true samples:")
        rng = default_rng(self.seed)
        theta = []
        # first sample an integer from [0, # Gaussians - 1], then sample from the corresponding Gaussian
        for _ in progress_bar(range(self.n)):
            i = rng.integers(0, len(self.mus))
            theta.append(multivariate_laplace.rvs(self.mus[i], self.Sigmas[i], 1).reshape(-1))
        return np.array(theta)

    ## Unadjusted Langevin Algorithm (ULA)
    def ula(self, gamma):
        print("\nSampling with ULA:")
        rng = default_rng(self.seed)
        theta0 = rng.normal(0, 1, self.d)
        theta = []
        # gammas = [gamma * i ** (-0.5) for i in range(1, self.n+1)]
        gammas = gamma * np.ones(self.n)
        for i in progress_bar(range(self.n)):
            xi = rng.multivariate_normal(np.zeros(self.d), np.eye(self.d))
            gamma = gammas[i]
            theta_new = self.gd_update(theta0, gamma) + np.sqrt(2*gamma) * xi
            theta.append(theta_new)    
            theta0 = theta_new
        return np.array(theta)


    ## Metropolis-Adjusted Langevin Algorithm (MALA)
    def q_prob(self, theta1, theta2, gamma):
        return multivariate_normal(mean=self.gd_update(theta2, gamma), cov=2*gamma).pdf(theta1)

    def prob(self, theta_new, theta_old, gamma):
        density_ratio = self.smooth_density_laplacian_mixture(theta_new) / self.smooth_density_laplacian_mixture(theta_old)
        q_ratio = self.q_prob(theta_old, theta_new, gamma) / self.q_prob(theta_new, theta_old, gamma)
        return density_ratio * q_ratio

    def mala(self, gamma):
        print("\nSampling with MALA:")
        rng = default_rng(self.seed)
        theta0 = rng.normal(0, 1, self.d)
        theta = []
        # gammas = [gamma * i ** (-0.5) for i in range(1, self.n+1)]
        gammas = gamma * np.ones(self.n)
        for i in progress_bar(range(self.n)):
            xi = rng.multivariate_normal(np.zeros(self.d), np.eye(self.d))
            gamma = gammas[i]
            theta_new = self.gd_update(theta0, gamma) + np.sqrt(2*gamma) * xi
            p = self.prob(theta_new, theta0, gamma)
            alpha = min(1, p)
            if rng.random() <= alpha:
                theta.append(theta_new)    
                theta0 = theta_new
        return np.array(theta), len(theta)


    ## Preconditioned ULA 
    def preconditioned_gd_update(self, theta, gamma, M): 
        return theta - gamma * M @ self.grad_smooth_potential_laplacian_mixture(theta)

    def pula(self, gamma, M):
        print("\nSampling with Preconditioned Langevin Algorithm:")
        rng = default_rng(self.seed)
        theta0 = rng.normal(0, 1, self.d)
        theta = []
        gammas = gamma * np.ones(self.n)
        for i in progress_bar(range(self.n)):
            xi = rng.multivariate_normal(np.zeros(self.d), np.eye(self.d))
            gamma = gammas[i]
            theta_new = self.preconditioned_gd_update(theta0, gamma, M) + np.sqrt(2*gamma) * sqrtm(M) @ xi
            theta.append(theta_new)    
            theta0 = theta_new
        return np.array(theta)


    # Preconditioning with inverse Hessian
    def ihpula(self, gamma):
        print("\nSampling with Inverse Hessian Preconditioned Unadjusted Langevin Algorithm:")
        rng = default_rng(self.seed)
        theta0 = rng.normal(0, 1, self.d)
        theta = []
        gammas = gamma * np.ones(self.n)
        for i in progress_bar(range(self.n)):
            xi = rng.multivariate_normal(np.zeros(self.d), np.eye(self.d))
            gamma = gammas[i]
            hess = self.hess_smooth_potential_laplacian_mixture(theta0)
            if len(self.mus) > 1:
                e = np.linalg.eigvals(hess)
                M = hess + (np.abs(min(e)) + .02) * np.eye(self.d)
                M = np.linalg.inv(M)
            else:
                M = np.linalg.inv(hess)
            theta_new = self.preconditioned_gd_update(theta0, gamma, M) + np.sqrt(2*gamma) * sqrtm(M) @ xi
            theta.append(theta_new)    
            theta0 = theta_new
        return np.array(theta)


    ## Mirror-Langevin Algorithm (MLA)
    def grad_mirror_hyp(self, theta, beta): 
        return np.arcsinh(theta / beta)

    def grad_conjugate_mirror_hyp(self, theta, beta):
        return beta * np.sinh(theta)

    def mla(self, gamma, beta):
        print("\nSampling with MLA: ")
        rng = default_rng(self.seed)
        theta0 = rng.normal(0, 1, self.d)
        theta = []
        gammas = gamma * np.ones(self.n)
        for i in progress_bar(range(self.n)):
            xi = rng.multivariate_normal(np.zeros(self.d), np.eye(self.d))
            gamma = gammas[i]
            theta_new = self.grad_mirror_hyp(theta0, beta) - gamma * self.grad_smooth_potential_laplacian_mixture(theta0) + np.sqrt(2*gamma) * (theta0**2 + beta**2)**(-.25) * xi
            theta_new = self.grad_conjugate_mirror_hyp(theta_new, beta)
            theta.append(theta_new)    
            theta0 = theta_new
        return np.array(theta)


## Main function
def lmc_laplacian_mixture(gamma_ula=5e-2, gamma_mala=5e-2, 
                         gamma_pula=5e-2, gamma_mla=5e-2, 
                         lamda=1e-1, alpha=1e-1, n=5, K=5000, seed=0):
    # Our 2-dimensional distribution will be over variables X and Y
    N = 300
    X = np.linspace(-5, 5, N)
    Y = np.linspace(-5, 5, N)
    X, Y = np.meshgrid(X, Y)

    # location parameters
    mu1 = np.array([0., 0.])
    mu2 = np.array([-2., 3.])
    mu3 = np.array([2., -3.])
    mu4 = np.array([3., 3.])
    mu5 = np.array([-2., -2.])

    if n == 1:
        mus = [mu1]
    elif n == 2: 
        mus = [mu1, mu2]
    elif n == 3: 
        mus = [mu1, mu2, mu3]
    elif n == 4: 
        mus = [mu2, mu3, mu4, mu5]
    elif n == 5: 
        mus = [mu1, mu2, mu3, mu4, mu5]

    # scale parameters
    alphas = np.ones(n) * alpha

    # weight vector
    omegas = np.ones(n) / n

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    lmc_laplacian = LangevinMonteCarloLaplacian(mus, alphas, omegas, lamda, K, seed)

    # The distribution on the variables X, Y packed into pos.
    Z = lmc_laplacian.density_laplacian_mixture(pos)
    Z_smooth = lmc_laplacian.smooth_density_laplacian_mixture(pos)


    ## Plot of the true Laplacian mixture
    print("\nPlotting the true Laplacian mixture...")
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
    # plt.pause(5)
    # plt.close()
    fig.savefig(f'./fig/fig_laplace_n{n}_gamma{gamma_ula}_lambda{lamda}_{K}_1.pdf', dpi=500)


    print("\nPlotting the smoothed Laplacian mixture...")    
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

    # plt.suptitle("2D Smoothed Laplacian Mixture") 
    plt.show(block=False)
    plt.pause(5)
    plt.close()
    fig.savefig(f'./fig/fig_laplace_n{n}_gamma{gamma_ula}_lambda{lamda}_{K}_1_smooth.pdf', dpi=500)

    Z2 = lmc_laplacian.ula(gamma_ula)    

    Z3, eff_K = lmc_laplacian.mala(gamma_mala)
    print(f'\nMALA percentage of effective samples: {eff_K / K} ')
        
    M = np.array([[1.0, 0.1], [0.1, 0.5]])
    Z4 = lmc_laplacian.pula(gamma_pula, M)        

    beta = np.array([0.7, 0.3])
    Z5 = lmc_laplacian.mla(gamma_mla, beta)


    ## Plot of the true Laplacian mixture with 2d histograms of samples
    print("\nConstructing the 2D histograms of samples...")
    fig3, axes = plt.subplots(2, 3, figsize=(13, 8))

    axes[0,0].contourf(X, Y, Z, cmap=cm.viridis)
    axes[0,0].set_title("True density", fontsize=16)

    axes[0,1].contourf(X, Y, Z_smooth, cmap=cm.viridis)
    axes[0,1].set_title("Smoothed density", fontsize=16)

    axes[0,2].hist2d(Z2[:,0], Z2[:,1], bins=100, cmap=cm.viridis)
    axes[0,2].set_title("ULA", fontsize=16)

    axes[1,0].hist2d(Z3[:,0], Z3[:,1], bins=100, cmap=cm.viridis)
    axes[1,0].set_title("MALA", fontsize=16)

    axes[1,1].hist2d(Z4[:,0], Z4[:,1], bins=100, cmap=cm.viridis)
    axes[1,1].set_title("PULA", fontsize=16)

    axes[1,2].hist2d(Z5[:,0], Z5[:,1], bins=100, cmap=cm.viridis)
    axes[1,2].set_title("MLA", fontsize=16)

    plt.show(block=False)
    plt.pause(5)
    plt.close()
    fig3.savefig(f'./fig/fig_laplace_n{n}_gamma{gamma_ula}_lambda{lamda}_{K}_3.pdf', dpi=500)  


    ## Plot of the true Laplacian mixture with KDE of samples
    print("\nConstructing the KDEs of samples...")
    fig2, axes = plt.subplots(2, 3, figsize=(13, 8))
    sns.set(font='serif', rc={'figure.figsize':(3.25, 3.5)})

    axes[0,0].contourf(X, Y, Z, cmap=cm.viridis)
    axes[0,0].set_title("True density", fontsize=16)

    axes[0,1].contourf(X, Y, Z_smooth, cmap=cm.viridis)
    axes[0,1].set_title("Smoothed density", fontsize=16)

    sns.kdeplot(x=Z2[:,0], y=Z2[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[0,2])
    axes[0,2].set_title("ULA", fontsize=16)

    sns.kdeplot(x=Z3[:,0], y=Z3[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[1,0])
    axes[1,0].set_title("MALA", fontsize=16)

    sns.kdeplot(x=Z4[:,0], y=Z4[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[1,1])
    axes[1,1].set_title("PULA", fontsize=16)

    sns.kdeplot(x=Z5[:,0], y=Z5[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[1,2])
    axes[1,2].set_title("MLA", fontsize=16)

    plt.show(block=False)
    plt.pause(5)
    plt.close()
    fig2.savefig(f'./fig/fig_laplace_n{n}_gamma{gamma_ula}_lambda{lamda}_{K}_2.pdf', dpi=600)
    fig2.savefig(f'./fig/fig_laplace_n{n}_gamma{gamma_ula}_lambda{lamda}_{K}_2.eps', dpi=1200)

    # Compute 2-Wasserstein distances between the samples generated by the algorithms and 
    # those generated directly from the true density
    ## Generate samples from the true density
    Z1 = lmc_laplacian.true_samples()

    ### 2-Wasserstein distances; see https://pythonot.github.io/auto_examples/plot_OT_2D_samples.html
    print("\nComputing 2-Wasserstein distances...")    
    # t0 = time.time()
    K1 = 10000
    Z1 = Z1[:K1]
    Z2 = Z2[:K1]
    Z3 = Z3[:K1]
    Z4 = Z4[:K1]
    Z5 = Z5[:K1]
    # M_ula = ot.dist(Z1, Z2)
    # M_mala = ot.dist(Z1, Z3)
    # M_pula = ot.dist(Z1, Z4)
    # M_mla = ot.dist(Z1, Z5)

    a = np.ones((K1,)) / K1
    b = np.ones((K1,)) / K1  # uniform distribution on samples
    # G0 = ot.emd(a, b, M_ula)
    # fig4 = plt.figure(figsize=(8, 8))
    # ot.plot.plot2D_samples_mat(Z1[:K1], Z2[:K1], G0, c=[.5, .5, 1])
    # plt.plot(Z1[:, 0], Z1[:, 1], '+b', label='True samples')
    # plt.plot(Z2[:, 0], Z2[:, 1], 'xr', label='ULA samples')
    # plt.legend(loc=0)
    # plt.title('OT matrix with samples')
    # plt.show(block=False)
    # plt.pause(20)
    # plt.close()

    # b_mala = np.ones((len(Z3),)) / len(Z3)
    # b_mala = np.ones((K1,)) / K1
    nitermax = int(1e5)
    # wass_ula = ot.emd2(a, b, M_ula, numItermax=nitermax, numThreads=16)
    # wass_mala = ot.emd2(a, b_mala, M_mala, numItermax=nitermax, numThreads=16)
    # wass_pula = ot.emd2(a, b, M_pula, numItermax=nitermax, numThreads=16)
    # wass_mla = ot.emd2(a, b, M_mla, numItermax=nitermax, numThreads=16)
    # print(f'2-Wasserstein distance between true samples and ULA samples: {wass_ula**.5}')
    # print(f'2-Wasserstein distance between true samples and MALA samples: {wass_mala**.5}')
    # print(f'2-Wasserstein distance between true samples and PULA samples: {wass_pula**.5}')
    # print(f'2-Wasserstein distance between true samples and MLA samples: {wass_mla**.5}')
    # t1 = time.time()
    # print(f'Time elapsed for computing 2-Wasserstein distances: {t1 - t0} seconds')
    
    
    print("\nComputing 2-Wasserstein distances vs samples...")
    wass_ula_list = []
    wass_mala_list = []
    wass_pula_list = []
    wass_mla_list = []

    interval = 100
    
    t0 = time.time()
    for k in progress_bar(range(1, K1)):
        if (k-1) % interval == 0:            
            b = np.ones((k+1,)) / (k+1)
            M_ula = ot.dist(Z1, Z2[:k+1,:])
            M_mala = ot.dist(Z1, Z3[:k+1,:])
            M_pula = ot.dist(Z1, Z4[:k+1,:])
            M_mla = ot.dist(Z1, Z5[:k+1,:])        
            wass_ula = ot.emd2(a, b, M_ula, numItermax=nitermax, numThreads=16)   
            wass_mala = ot.emd2(a, b, M_mala, numItermax=nitermax, numThreads=16)     
            wass_pula = ot.emd2(a, b, M_pula, numItermax=nitermax, numThreads=16)
            wass_mla = ot.emd2(a, b, M_mla, numItermax=nitermax, numThreads=16)
            wass_ula_list.append(wass_ula**.5)
            wass_mala_list.append(wass_mala**.5)
            wass_pula_list.append(wass_pula**.5)
            wass_mla_list.append(wass_mla**.5)
    t1 = time.time()
    print(f'\nTime elapsed for computing 2-Wasserstein distances: {t1 - t0} seconds')

    ## Plot of 2-Wasserstein distances vs samples
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.style.use(['science'])
    plt.rcParams.update({
        "font.family": "serif",   # specify font family here
        "font.serif": ["Times"],  # specify font here
        "text.usetex": True,
        "axes.prop_cycle": plt.cycler("color", plt.cm.tab10.colors),
        } 
    )

    fig3 = plt.figure(figsize=(6, 4))
    iters = [k+1 for k in range(1, K1) if (k-1) % interval == 0]
    plt.plot(iters, wass_ula_list, label='ULA')
    plt.plot(iters, wass_mala_list, label='MALA')
    plt.plot(iters, wass_pula_list, label='PULA')
    plt.plot(iters, wass_mla_list, label='MLA')
    plt.xlabel('sample')
    plt.ylabel(r'2-Wasserstein distance')
    plt.legend()
    plt.show(block=False)
    plt.pause(10)
    plt.close()
    fig3.savefig(f'./fig/fig_laplace_n{n}_gamma{gamma_ula}_lambda{lamda}_{K}_wass_dist.pdf', dpi=600)
    fig3.savefig(f'./fig/fig_laplace_n{n}_gamma{gamma_ula}_lambda{lamda}_{K}_wass_dist.eps', dpi=1200)


if __name__ == '__main__':
    if not os.path.exists('fig'):
        os.makedirs('fig')
    fire.Fire(lmc_laplacian_mixture)