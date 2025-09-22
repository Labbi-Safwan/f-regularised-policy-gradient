import numpy as np
import matplotlib.pyplot as plt
import math
import argparse
import os

from functools import partial
import chex
import jax.numpy as jnp
import jax
 
 
 
@partial(jax.jit, static_argnums=(2, 3))
def f_tsallis_softmax_jax(
    theta: chex.Array, prior: chex.Array, alpha: float, eps: float = 1e-6, max_iter: int = 30
) -> tuple[chex.Array, float]:
    """
    Compute the f-softmax of a vector theta using the provided functions f_prime and f_star_prime.
 
    This implementation assumes that f'(0) is infinite.
 
    Parameters:
    theta (np.ndarray): Input vector.
    prior (np.ndarray): Prior distribution.
    alpha (float): Tsallis entropy parameter.
    eps (float): Tolerance for convergence.
 
    Returns:
    tuple[np.ndarray, float]: The f-softargmax of the input vector theta and f-softmax
    """
    jax.debug.print("Start!")
    j_star = jnp.argmax(theta)
    theta_max = jnp.max(theta)
    prior_star = prior[j_star]
    tau_min = theta_max - ((1 / prior_star) ** (alpha - 1.0) - 1.0) / (alpha - 1.0)
    tau_max = theta_max - ((1 / np.sum(prior)) ** (alpha - 1.0) - 1.0) / (alpha - 1.0)
 
    tau = (tau_min + tau_max) / 2
    p_tau = prior * (1 + (alpha - 1.0) * (theta - tau)) ** (1.0 / (alpha - 1.0))
    phi_tau = jnp.sum(p_tau) - 1
 
    def cond_fun(state):
        iter, _, _, phi_tau = state
        return (jnp.abs(phi_tau) > eps) & (iter < max_iter)
 
    def body_fun(state):
        iter, tau_min, tau_max, phi_tau = state
        tau = (tau_min + tau_max) / 2.0
        # jax.lax.cond is a functional equivalent of if/else
        tau_min, tau_max = jax.lax.cond(
            phi_tau < 0,
            lambda: (tau_min, tau),
            lambda: (tau, tau_max),
        )
        tau = (tau_min + tau_max) / 2.0
        p_tau = prior * (1 + (alpha - 1.0) * (theta - tau)) ** (1.0 / (alpha - 1.0))
        phi_tau = jnp.sum(p_tau) - 1
        return iter+1, tau_min, tau_max, phi_tau
 
    # Initial state for the loop
    init_state = (0, tau_min, tau_max, phi_tau)
    # Run the bisection loop
    iter, tau_min, tau_max, _ = jax.lax.while_loop(cond_fun, body_fun, init_state)
    tau = (tau_min + tau_max) / 2
    p_tau = prior * (1 + (alpha - 1.0) * (theta - tau)) ** (1.0 / (alpha - 1.0))
 
    fsoftmax = tau + jnp.sum(
        prior * ((1 + (alpha - 1.0) * (theta - tau)) ** (alpha / (alpha - 1.0)) - 1.0) / alpha
    )
    jax.debug.print("end!")
    return p_tau, fsoftmax
 
 
def f_tsallis_softmax(theta, prior, alpha, eps=1e-6):
    jax_theta = jnp.array(theta)
    jax_prior = jnp.array(prior)
    print('before_jax')
    dist, fsoftmax = f_tsallis_softmax_jax(jax_theta, jax_prior, alpha, eps)
    print('after_jax')

    return np.array(dist), float(fsoftmax)

def softmax_policy(theta, prior):
    """Compute action probabilities."""
    logits = theta
    # Numerical stability: subtract max
    stable_logits = logits - np.max(logits)
    # Exponentiated logits, weighted by prior
    weighted_exp = prior * np.exp(stable_logits)
    # Normalize
    probs = weighted_exp / np.sum(weighted_exp)
    return probs

def compute_divergence_shannon(probs, prior):
    """Compute KL-divergence between two probabilities."""
    log_ratio = np.log(probs/prior)
    divergence = np.sum(probs*log_ratio)
    return divergence

def compute_divergence_tsallis(probs, prior, alpha):
    """Compute alpha-Tsallis divergence between two probabilities.."""
    u = probs / prior
    f_u = (np.power(u, alpha) - alpha * u + alpha - 1.0) / (alpha * (alpha - 1.0))
    divergence = np.sum(prior * f_u)
    return divergence


class fPG:
    def __init__(self, env , step_size, temp,  **kwargs):
        self.kwargs = kwargs
        # get the environments 
        self.env = env
        self.S = env.S
        self.A = env.A
        print(self.S, self.A)
        self.gamma = kwargs.get('discount')
        self.T = kwargs.get('n_iteration')
        self.H = kwargs.get('len_truncation')
        self.step = step_size
        self.temperature  = temp
        self.environnement = kwargs.get('environment')
        self.alpha = kwargs.get('alpha')
        self.B = kwargs.get('batch_size')
        # set number of agents
        self.verbose = kwargs.get('verbose')
        self.theta = np.zeros((self.S, self.A)) 
        self.init_dist = np.ones(self.S)/self.S
        self.prior = np.ones(self.A)/self.A


    def compute_policy(self, theta, state):
        """Compute action probabilities from theta for a given state."""
        logits = theta[state]
        if self.alpha ==1:
            probs = softmax_policy(logits, self.prior)
        else:
            print(self.alpha)
            probs = f_tsallis_softmax(logits, self.prior, self.alpha, eps=1e-6)
        return probs
    
    def sample_action(self, probs):
        """Sample an action according to given action probabilities."""
        return np.random.choice(len(probs), p=probs)

    def compute_divergence(self, probs):
        if self.alpha ==1:
            divergence = compute_divergence_shannon(probs, self.prior)
        else:
            divergence = compute_divergence_tsallis(probs, self.prior, self.alpha)
        return divergence
    
    def compute_grad_log_pi(self, probs, state, action):
        if self.alpha ==1:
            grad = np.zeros((self.S, self.A))           # initialize (S,A) matrix
            grad[state, :] = -probs                      # for all actions at that state
            grad[state, action] += 1                     # plus 1 at (state, action)
        else:
            ref_probs = self.prior    
            u = probs / ref_probs
            fpp = u ** (self.alpha - 2)

            # normalizing denominator W^f(s)
            W = np.sum(ref_probs * fpp)

            # weights w^f_θ(a|s)
            w = fpp / W

            grad = np.zeros((self.S, self.A))

            # Now implement the formula:
            # ∂ log π^f(a|s)/∂θ(s,b) = 1_{s'=s} * W^f(s)/π^f(a|s) * [1_b(a) w(a|s) - w(a|s) w(b|s)]
            factor = W / probs[action]
            for b in range(self.A):
                grad[state, b] = factor * ( (1 if b == action else 0) * w[action] - w[action] * w[b] )                   # plus 1 at (state, action)
        return grad
    
    def compute_vector_F(self, probs, state):
        if self.alpha < 1:
            ref_probs = self.prior    
            u = probs / ref_probs
            fpp = u ** (self.alpha - 2)

            # normalizing denominator W^f(s)
            W = np.sum(ref_probs * fpp)

            # weights w^f_θ(a|s)
            w = fpp / W

            F = np.zeros((self.S, self.A))

            # Now implement the formula:
            # ∂ log π^f(a|s)/∂θ(s,b) = 1_{s'=s} * W^f(s)/π^f(a|s) * [1_b(a) w(a|s) - w(a|s) w(b|s)]
            f_u_prime = (np.power(u, self.alpha-1) - 1.0) / (self.alpha - 1.0)
            divergence_f_prime = np.sum(w * f_u_prime)
            for b in range(self.A):
                F[state, b] = W * fpp[b] * ( f_u_prime[b] - divergence_f_prime )    
        else:
            ref_probs = self.prior    
            u = probs / ref_probs
            fpp = u ** (- 1)

            # normalizing denominator W^f(s)
            W = np.sum(ref_probs * fpp)

            # weights w^f_θ(a|s)
            w = fpp / W

            F = np.zeros((self.S, self.A))

            # Now implement the formula:
            # ∂ log π^f(a|s)/∂θ(s,b) = 1_{s'=s} * W^f(s)/π^f(a|s) * [1_b(a) w(a|s) - w(a|s) w(b|s)]
            f_u_prime = np.log(u)
            divergence_f_prime = np.sum(w * f_u_prime)
            for b in range(self.A):
                F[state, b] = W * fpp[b] * ( f_u_prime[b] - divergence_f_prime )         
        return F

    def train(self):
        #minimal_probability = []
        theta = np.zeros((self.S, self.A))
        true_objectives = []
        for t in range(self.T):
            policy = np.zeros((self.S, self.A))
            for state in range(self.S):
                policy_state = self.compute_policy(theta, state)[0]
                policy_state = np.clip(policy_state, 0, None)
                policy_state /= policy_state.sum()
                policy[state,:] = policy_state
            
            avg_return = self.compute_objective(policy)
            true_objectives.append(avg_return)

            #minimal_probability_iteration = np.min(policy)
            #minimal_probability.append(minimal_probability_iteration)
            if self.verbose:
                print('Iteration Number:',t)
                print('Policy:', policy)
                print('Theta:', theta)
                print('Return:', avg_return)
            #print('agent',m)
            env = self.env
            trajectories = []
            for _ in range(self.B): 
                state = env.reset()
                trajectory = []
                for _ in range(self.H):
                    probs = policy[state,:]
                    probs = probs/probs.sum()
                    action = self.sample_action(probs)
                    next_state, reward = env.step(action)
                    trajectory.append((state, action, reward))
                    state = next_state
                trajectories.append(trajectory)

            # Now, estimate the gradient over the whole batch
            grads = np.zeros((self.S, self.A))

            for trajectory in trajectories:
                cumulative_grad_log_pi = np.zeros((self.S, self.A))

                for t, (s_t, a_t, r_t) in enumerate(trajectory):
                    # Accumulate gradient sum up to time t
                    grad_log_pi_t = self.compute_grad_log_pi(policy[s_t,:], s_t, a_t)
                    cumulative_grad_log_pi += grad_log_pi_t

                    # Add contribution to total gradient
                    divergence = self.compute_divergence(policy[s_t,:])
                    grads += (self.gamma ** t) * cumulative_grad_log_pi * r_t
                    grads -= self.temperature * (self.gamma ** t) * cumulative_grad_log_pi * divergence
                    function_F = self.compute_vector_F(policy[s_t,:],s_t)
                    grads += (self.gamma ** t) * function_F

            # Average over B trajectories
            grads /= self.B
            #print(grads)

            # Update theta
            theta += self.step * grads

        return true_objectives #, minimal_probability

    def compute_mrp_transition(self, policy):
        transition_kernel =  self.env.get_P()
        mrp_transition = np.sum(policy[:, :, np.newaxis] * transition_kernel, axis=1)
        return mrp_transition

    def compute_mrp_reward(self, policy):
        reward = self.env.get_r()
        # Element-wise multiplication and sum along the actions axis
        mrp_reward = np.sum(policy * reward, axis=1)
        return mrp_reward

    def compute_stationnary_distribution(self, policy):
        mrp_transition = self.compute_mrp_transition( policy)
        stationnary_distribution = (1- self.gamma) * self.init_dist.T @ np.linalg.inv(np.eye(self.S) - self.gamma *mrp_transition)
        return stationnary_distribution

    def compute_value_function(self, policy):
        mrp_transition = self.compute_mrp_transition(policy)
        mrp_reward = self.compute_mrp_reward(policy)
        return np.linalg.inv(np.eye(self.S) -self.gamma *mrp_transition) @ mrp_reward

    def compute_qfunction(self, policy):
        reward = self.env.get_r()
        transitions = self.env.get_P()
        value_function = self.compute_value_function(policy)
        expected_future_rewards = np.sum(transitions * value_function[np.newaxis, np.newaxis, :], axis=2)
        Q_function = reward + self.gamma * expected_future_rewards
        return Q_function

    def compute_objective(self,policy):
        statinnary_distrubtion_agent = self.compute_stationnary_distribution(policy)
        reward_mrp = self.compute_mrp_reward(policy)
        objective = (1 / (1 - self.gamma))*np.dot(statinnary_distrubtion_agent,reward_mrp)
        return(objective)