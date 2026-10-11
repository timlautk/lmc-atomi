# Copyright 2023 by Tim Tsz-Kit Lau
# MIT License

# To install JAX, see its documentations
# Install libraries: pip install -U numpy matplotlib scipy seaborn fire fastprogress SciencePlots scikit-image pylops pyproximal jax blackjax optax

import os
import itertools
from fastprogress import progress_bar
from typing import NamedTuple
import fire

import numpy as np
from scipy.stats import gaussian_kde

import jax
from jax import grad, jit
import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy.stats as stats

import blackjax
import blackjax.sgmcmc.gradients as gradients
import optax
from blackjax.types import PyTree
from optax._src.base import OptState

import prox_jax 
import prox_tv as ptv

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
sns.set(font="Times")


class GaussianMixturewithTVPriorSampling:
    def __init__(self, lamda, alpha, positions, sigma) -> None:
        self.lamda = lamda 
        self.alpha = alpha
        self.positions = positions
        self.mu = jnp.array([list(prod) for prod in itertools.product(positions, positions)])
        self.sigma = sigma * jnp.eye(2)

    def logprob_fn(self, x, *_):
        return self.lamda * jsp.special.logsumexp(stats.multivariate_normal.logpdf(x, self.mu, self.sigma))
    
    def logprior_fn(self, x, *_):
        return stats.laplace.logpdf(x, scale=self.alpha)

    def sample_fn(self, rng_key):
        choose_key, sample_key = jax.random.split(rng_key)
        samples = jax.random.multivariate_normal(sample_key, self.mu, self.sigma)
        return jax.random.choice(choose_key, samples)

    def sampling(self, seed, xmin, ymin, xmax, ymax, nbins):
        rng_key = jax.random.PRNGKey(seed)
        samples = jax.vmap(self.sample_fn)(jax.random.split(rng_key, 10_000))

        x, y = samples[:, 0], samples[:, 1]
        xx, yy = np.mgrid[xmin:xmax:nbins, ymin:ymax:nbins]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])
        kernel = gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)

        '''
        fig, ax = plt.subplots()
        cfset = ax.contourf(xx, yy, f, cmap=cm.viridis)
        ax.imshow(np.rot90(f), cmap=cm.viridis, extent=[xmin, xmax, ymin, ymax])
        cset = ax.contour(xx, yy, f, colors='k')

        plt.rcParams['axes.titlepad'] = 15.
        plt.title("Samples from a mixture of 25 normal distributions")

        plt.show(block=False)
        plt.pause(5)
        plt.close()
        '''

        print("Constructing the true 2D Gaussian mixture density... ")
        fig2 = plt.figure(figsize=(10, 5))
        ax1 = fig2.add_subplot(1, 2, 1, projection='3d')

        ax1.plot_surface(xx, yy, f, rstride=3, cstride=3, linewidth=1, antialiased=True, cmap=cm.viridis)
        ax1.view_init(45, -70)

        ax2 = fig2.add_subplot(1, 2, 2, projection='3d')
        ax2.contourf(xx, yy, f, zdir='z', offset=0, cmap=cm.viridis)
        ax2.view_init(90, 270)

        ax2.grid(False)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_zticks([])

        plt.suptitle("True 2D Gaussian Mixture") 
        plt.show(block=False)
        plt.pause(10)
        plt.close()
        return f


class SPGLD:
    def __init__(self, lamda, alpha, positions, sigma) -> None:
        self.lamda = lamda 
        self.alpha = alpha
        self.positions = positions
        self.mu = jnp.array([list(prod) for prod in itertools.product(positions, positions)])
        self.sigma = sigma * jnp.eye(2)

    def logprob_fn(self, x, *_):
        return self.lamda * jsp.special.logsumexp(jax.scipy.stats.multivariate_normal.logpdf(x, self.mu, self.sigma))
    
    def logprior_fn(self, x, *_):
        return stats.laplace.logpdf(x, scale=self.alpha)

    def sampling(self, seed=0, num_training_steps=50000):
        data_size = 1000
        schedule_fn = lambda k: 0.05 * k ** (-0.55)
        schedule = [schedule_fn(i) for i in range(1, num_training_steps+1)]

        # grad_fn = lambda x, _: jax.grad(self.logprob_fn)(x)
        grad_fn = gradients.grad_estimator(self.logprior_fn, self.logprob_fn, data_size)
        spgld = blackjax.sgld(grad_fn)        

        rng_key = jax.random.PRNGKey(seed)
        init_position = -10 + 20 * jax.random.uniform(rng_key, shape=(2,))

        position = init_position
        spgld_samples = []

        print("\nSampling with SGLD:")
        for i in progress_bar(range(num_training_steps)):
            _, rng_key = jax.random.split(rng_key)
            position = jax.jit(spgld)(rng_key, position, 0, schedule[i])
            spgld_samples.append(position)

        '''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x = [sample[0] for sample in sgld_samples]
        y = [sample[1] for sample in sgld_samples]

        ax.plot(x, y, 'k-', lw=0.1, alpha=0.5)
        ax.set_xlim([-8, 8])
        ax.set_ylim([-8, 8])

        plt.axis('off')
        # plt.show()
        plt.show(block=False)
        plt.pause(5)
        plt.close()
        '''
        return np.array(spgld_samples)


class SSGLD:
    def __init__(self, lamda, alpha, positions, sigma) -> None:
        self.lamda = lamda 
        self.alpha = alpha
        self.positions = positions
        self.mu = jnp.array([list(prod) for prod in itertools.product(positions, positions)])
        self.sigma = sigma * jnp.eye(2)

    def logprob_fn(self, x, *_):
        return self.lamda * jsp.special.logsumexp(jax.scipy.stats.multivariate_normal.logpdf(x, self.mu, self.sigma))
    
    def logprior_fn(self, x, *_):
        return stats.laplace.logpdf(x, scale=self.alpha)

    def sampling(self, seed=0, num_training_steps=50000):
        data_size = 100
        schedule_fn = lambda k: 0.05 * k ** (-0.55)
        schedule = [schedule_fn(i) for i in range(1, num_training_steps+1)]

        # grad_fn = lambda x, _: jax.grad(self.logprob_fn)(x)
        grad_fn = gradients.grad_estimator(self.logprior_fn, self.logprob_fn, data_size)
        ssgld = blackjax.sgld(grad_fn)        

        rng_key = jax.random.PRNGKey(seed)
        init_position = -10 + 20 * jax.random.uniform(rng_key, shape=(2,))

        position = init_position
        ssgld_samples = []

        print("\nSampling with SGLD:")
        for i in progress_bar(range(num_training_steps)):
            _, rng_key = jax.random.split(rng_key)
            position = jax.jit(ssgld)(rng_key, position, 0, schedule[i])
            ssgld_samples.append(position)

        '''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x = [sample[0] for sample in sgld_samples]
        y = [sample[1] for sample in sgld_samples]

        ax.plot(x, y, 'k-', lw=0.1, alpha=0.5)
        ax.set_xlim([-8, 8])
        ax.set_ylim([-8, 8])

        plt.axis('off')
        # plt.show()
        plt.show(block=False)
        plt.pause(5)
        plt.close()
        '''
        return np.array(ssgld_samples)



class MYSGLD:
    def __init__(self, lamda, alpha, gamma, positions, sigma) -> None:
        self.lamda = lamda 
        self.alpha = alpha
        self.gamma = gamma
        self.positions = positions
        self.mu = jnp.array([list(prod) for prod in itertools.product(positions, positions)])
        self.sigma = sigma * jnp.eye(2)

    def logprob_fn(self, x, *_):
        return self.lamda * jsp.special.logsumexp(jax.scipy.stats.multivariate_normal.logpdf(x, self.mu, self.sigma))
    
    def logprior_fn(self, x, *_):
        return stats.laplace.logpdf(x, scale=self.alpha)
    
    def grad_Moreau_env(self, x, *_):
        return (self.gamma - prox_jax.prox_laplace(x, self.gamma * self.alpha)) / self.gamma

    def prox_update(self, x, *_):
        return -self.gamma * self.grad_Moreau_env(x, self.lamda, self.alpha)

    def sampling(self, seed=0, num_training_steps=50000):
        data_size = 100
        schedule_fn = lambda k: 0.05 * k ** (-0.55)
        schedule = [schedule_fn(i) for i in range(1, num_training_steps+1)]

        # grad_fn = lambda x, _: jax.grad(self.logprob_fn)(x)
        grad_fn = gradients.grad_estimator(self.logprior_fn, self.logprob_fn, data_size)
        ssgld = blackjax.sgld(grad_fn)        

        rng_key = jax.random.PRNGKey(seed)
        init_position = -10 + 20 * jax.random.uniform(rng_key, shape=(2,))

        position = init_position
        ssgld_samples = []

        print("\nSampling with SGLD:")
        for i in progress_bar(range(num_training_steps)):
            _, rng_key = jax.random.split(rng_key)
            position = jax.jit(ssgld)(rng_key, position, 0, schedule[i])
            ssgld_samples.append(position)

        '''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x = [sample[0] for sample in sgld_samples]
        y = [sample[1] for sample in sgld_samples]

        ax.plot(x, y, 'k-', lw=0.1, alpha=0.5)
        ax.set_xlim([-8, 8])
        ax.set_ylim([-8, 8])

        plt.axis('off')
        # plt.show()
        plt.show(block=False)
        plt.pause(5)
        plt.close()
        '''
        return np.array(ssgld_samples)


class ScheduleState(NamedTuple):
    step_size: float
    do_sample: bool

def build_schedule(num_training_steps, num_cycles=4, initial_step_size=1e-3, exploration_ratio=0.25):
    cycle_length = num_training_steps // num_cycles

    def schedule_fn(step_id):
        do_sample = False
        if ((step_id % cycle_length)/cycle_length) >= exploration_ratio:
            do_sample = True

        cos_out = jnp.cos(jnp.pi * (step_id % cycle_length) / cycle_length) + 1
        step_size = 0.5 * cos_out * initial_step_size

        return ScheduleState(step_size, do_sample)

    return schedule_fn


class CyclicalSGMCMCState(NamedTuple):
    """State of the Cyclical SGMCMC sampler.
    """
    position: PyTree
    opt_state: OptState


def cyclical_sgld(grad_estimator_fn, loglikelihood_fn):
    # Initialize the SgLD step function
    sgld = blackjax.sgld(grad_estimator_fn)
    sgd = optax.sgd(1.)

    def init_fn(position):
        opt_state = sgd.init(position)
        return CyclicalSGMCMCState(position, opt_state)

    def step_fn(rng_key, state, minibatch, schedule_state):
        """Cyclical SGLD kernel."""

        def step_with_sgld(current_state):
            rng_key, state, minibatch, step_size = current_state
            new_position = sgld(rng_key, state.position, minibatch, step_size)
            return CyclicalSGMCMCState(new_position, state.opt_state)

        def step_with_sgd(current_state):
            _, state, minibatch, step_size = current_state
            grads = grad_estimator_fn(state.position, 0)
            rescaled_grads = - 1. * step_size * grads
            updates, new_opt_state = sgd.update(rescaled_grads, state.opt_state, state.position)
            new_position = optax.apply_updates(state.position, updates)
            return CyclicalSGMCMCState(new_position, new_opt_state)

        new_state = jax.lax.cond(
            schedule_state.do_sample,
            step_with_sgld,
            step_with_sgd,
            (rng_key, state, minibatch, schedule_state.step_size)
        )

        return new_state

    return init_fn, step_fn


class cyclicalSPGLD:
    def __init__(self, lamda, alpha, positions, sigma) -> None:
        self.lamda = lamda 
        self.alpha = alpha
        self.positions = positions
        self.mu = jnp.array([list(prod) for prod in itertools.product(positions, positions)])
        self.sigma = sigma * jnp.eye(2)

    def logprob_fn(self, x, *_):
        return self.lamda * jsp.special.logsumexp(jax.scipy.stats.multivariate_normal.logpdf(x, self.mu, self.sigma))
    
    def logprior_fn(self, x, *_):
        return stats.laplace.logpdf(x, scale=self.alpha)

    def sampling(self, seed=0, num_training_steps=50000):       
        data_size = 1000 
        schedule_fn = build_schedule(num_training_steps, 30, 0.09, 0.25)
        schedule = [schedule_fn(i) for i in range(num_training_steps)]

        # grad_fn = lambda x, _: jax.grad(self.logprob_fn)(x)
        grad_fn = gradients.grad_estimator(self.logprior_fn, self.logprob_fn, data_size)
        init, step = cyclical_sgld(grad_fn, self.logprob_fn)

        rng_key = jax.random.PRNGKey(seed)
        init_position = -10 + 20 * jax.random.uniform(rng_key, shape=(2,))
        init_state = init(init_position)

        state = init_state
        cyclical_samples = []
        print("\nSampling with Cyclical SPGLD:")
        for i in progress_bar(range(num_training_steps)):
            _, rng_key = jax.random.split(rng_key)
            state = jax.jit(step)(rng_key, state, 0, schedule[i])
            if schedule[i].do_sample:
                cyclical_samples.append(state.position)
        return np.array(cyclical_samples)


class cyclicalSSGLD:
    def __init__(self, lamda, alpha, positions, sigma) -> None:
        self.lamda = lamda 
        self.alpha = alpha
        self.positions = positions
        self.mu = jnp.array([list(prod) for prod in itertools.product(positions, positions)])
        self.sigma = sigma * jnp.eye(2)

    def logprob_fn(self, x, *_):
        return self.lamda * jsp.special.logsumexp(jax.scipy.stats.multivariate_normal.logpdf(x, self.mu, self.sigma))

    def logprior_fn(self, x, *_):
        return stats.laplace.logpdf(x, scale=self.alpha)
    
    def sampling(self, seed=0, num_training_steps=50000):  
        data_size = 1000      
        schedule_fn = build_schedule(num_training_steps, 30, 0.09, 0.25)
        schedule = [schedule_fn(i) for i in range(num_training_steps)]

        # grad_fn = lambda x, _: jax.grad(self.logprob_fn)(x)
        grad_fn = gradients.grad_estimator(self.logprior_fn, self.logprob_fn, data_size)
        init, step = cyclical_sgld(grad_fn, self.logprob_fn)

        rng_key = jax.random.PRNGKey(seed)
        init_position = -10 + 20 * jax.random.uniform(rng_key, shape=(2,))
        init_state = init(init_position)

        state = init_state
        cyclical_samples = []
        print("\nSampling with Cyclical SSGLD:")
        for i in progress_bar(range(num_training_steps)):
            _, rng_key = jax.random.split(rng_key)
            state = jax.jit(step)(rng_key, state, 0, schedule[i])
            if schedule[i].do_sample:
                cyclical_samples.append(state.position)
        return np.array(cyclical_samples)


class contourSPGLD:
    def __init__(self, lamda, alpha, positions, sigma) -> None:
        self.lamda = lamda 
        self.alpha = alpha
        self.positions = positions
        self.mu = jnp.array([list(prod) for prod in itertools.product(positions, positions)])
        self.sigma = sigma * jnp.eye(2)

    def logprob_fn(self, x, *_):
        return self.lamda * jsp.special.logsumexp(jax.scipy.stats.multivariate_normal.logpdf(x, self.mu, self.sigma))

    def logprior_fn(self, x, *_):
        return stats.laplace.logpdf(x, scale=self.alpha)
    
    def sample_fn(self, rng_key, num_samples):
        _, sample_key = jax.random.split(rng_key)
        samples = jax.random.multivariate_normal(sample_key, self.mu, self.sigma, shape=(num_samples, self.mu.shape[0],))
        return samples

    def sampling(self, zeta, sz, lr=1e-3, temperature=50, num_partitions=100000, energy_gap=0.25, domain_radius=50, seed=0, num_training_steps=50000):
        data_size = 1000
        batch_size = 1000

        rng_key = jax.random.PRNGKey(seed)
        rng_key, sample_key = jax.random.split(rng_key)
        X_data = self.sample_fn(sample_key, data_size)

        # logprior_fn = lambda _: 0
        logdensity_fn = gradients.logdensity_estimator(self.logprior_fn, self.logprob_fn, data_size)
        csgld = blackjax.csgld(
                    logdensity_fn,
                    # self.logprob_fn,
                    zeta=zeta,  # can be specified at each step in lower-level interface
                    temperature=temperature,  # can be specified at each step
                    num_partitions=num_partitions,  # cannot be specified at each step
                    energy_gap=energy_gap,  # cannot be specified at each step
                    min_energy=0,
                )

        # rng_key = jax.random.PRNGKey(seed)
        init_position = -10 + 20 * jax.random.uniform(rng_key, shape=(2,))
        state = csgld.init(init_position)
        csgld_samples, csgld_energy_idx_list = [], jnp.array([])

        print("\nSampling with Contour SGLD:")
        for i in progress_bar(range(num_training_steps)):
            # rng_key, subkey = jax.random.split(rng_key)
            _, rng_key = jax.random.split(rng_key)
            stepsize_SA = min(1e-2, (i + 100) ** (-0.8)) * sz
            # data_batch = jax.random.shuffle(rng_key, X_data)[:batch_size, :, :]
            data_batch = X_data
            state = jax.jit(csgld.step)(rng_key, state, data_batch, lr, stepsize_SA)
            # state = jax.jit(csgld.step)(subkey, state, 0, lr, stepsize_SA)
            csgld_samples.append(state.position)
            # csgld_samples = jnp.append(csgld_samples, state.position)
            csgld_energy_idx_list = jnp.append(csgld_energy_idx_list, state.energy_idx)
        
        csgld_samples = jnp.array(csgld_samples)

        important_idx = jnp.where(state.energy_pdf > jnp.quantile(state.energy_pdf, 0.95))[0]
        scaled_energy_pdf = state.energy_pdf[important_idx] ** zeta / (state.energy_pdf[important_idx] ** zeta).max()

        csgld_re_samples = []
        print("\nResampling:")
        for _ in progress_bar(range(5)):
            rng_key, _ = jax.random.split(rng_key)
            for my_idx in important_idx:
                if jax.random.bernoulli(rng_key, p=scaled_energy_pdf[my_idx], shape=None) == 1:
                    samples_in_my_idx = csgld_samples[csgld_energy_idx_list == my_idx]
                    csgld_re_samples.extend(samples_in_my_idx)
        return np.array(csgld_re_samples)


class ULPDA:
    def __init__(self, lamda, alpha, positions, sigma) -> None:
        self.lamda = lamda 
        self.alpha = alpha
        self.positions = positions
        self.mu = jnp.array([list(prod) for prod in itertools.product(positions, positions)])
        self.sigma = sigma * jnp.eye(2)

    def logprob_fn(self, x, *_):
        return self.lamda * jsp.special.logsumexp(jax.scipy.stats.multivariate_normal.logpdf(x, self.mu, self.sigma))

    def logprior_fn(self, x, *_):
        return stats.laplace.logpdf(x, scale=self.alpha)
    
    def sample_fn(self, rng_key, num_samples):
        _, sample_key = jax.random.split(rng_key)
        samples = jax.random.multivariate_normal(sample_key, self.mu, self.sigma, shape=(num_samples, self.mu.shape[0],))
        return samples





    

def main(lamda=1/25, zeta=.75, sz=10, lr=1e-3, temp=1, num_partitions=50, seed=0, num_training_steps=50000, n=10000):
    positions = [-4, -2, 0, 2, 4]
    sigma = 0.03
    xmin, ymin = -5, -5
    xmax, ymax = 5, 5
    N = 300
    nbins = 300j

    X = np.linspace(-5, 5, N)
    Y = np.linspace(-5, 5, N)
    X, Y = np.meshgrid(X, Y)
    
    Z = GaussianMixturewithTVPriorSampling(lamda, alpha, positions, sigma).sampling(seed, xmin, ymin, xmax, ymax, nbins)

    Z2 = SPGLD(lamda, positions, sigma).sampling(seed, num_training_steps)

    Z3 = SSGLD().sampling(seed, num_training_steps)

    Z4 = MYSGLD().sampling(seed, num_training_steps)

    Z5 = cyclicalSPGLD(lamda, positions, sigma).sampling(seed, num_training_steps)

    Z6 = cyclicalSSGLD(lamda, positions, sigma).sampling(seed, num_training_steps)

    # zeta = 0.75
    # sz = 10
    # lr = 1e-3
    # temp = 1
    # num_partitions = 50
    energy_gap = 0.25
    domain_radius = 50
    # n = 10000
    Z4 = contourSGLD(lamda, positions, sigma).sampling(zeta, sz, lr, temp, num_partitions, energy_gap, domain_radius, seed, n)

    # Z5 = HMC().sampling()


    
    print("\nConstructing the plots of samples...")
    fig2, axes = plt.subplots(2, 3, figsize=(13, 8))
    # fig2.suptitle("True density and KDEs of samples") 

    axes[0,0].contourf(X, Y, Z, cmap=cm.viridis)
    axes[0,0].set_title("True density", fontsize=16)

    sns.kdeplot(x=Z2[:,0], y=Z2[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[0,1])
    axes[0,1].set_title("SPGLD", fontsize=16)

    sns.kdeplot(x=Z3[:,0], y=Z3[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[0,2])
    axes[0,2].set_title("SSGLD", fontsize=16)

    
    sns.kdeplot(x=Z4[:,0], y=Z4[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[1,0])
    axes[1,0].set_title("MYSGLD", fontsize=16)

    sns.kdeplot(x=Z5[:,0], y=Z5[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[1,1])
    axes[1,1].set_title("Cylical SPGLD", fontsize=16)

    sns.kdeplot(x=Z6[:,0], y=Z6[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[1,2])
    axes[1,2].set_title("Cylical SSGLD", fontsize=16)

    plt.show()
    # plt.pause(5)
    # plt.close()
    fig2.savefig(f'./fig/fig_prox_sgld_{n}_2.pdf', dpi=500)  



if __name__ == '__main__':
    if not os.path.exists('fig'):
        os.makedirs('fig')
    fire.Fire(main)

    



