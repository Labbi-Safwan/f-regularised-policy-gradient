import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import pickle

from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
import gymnax
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper

from functools import partial
from itertools import product
from tqdm import tqdm

from fdiv import AlphaDivergence


class ActorCritic(nn.Module):
    """Actor-Critic network with Tsallis-softmax parametrisation."""
    action_dim: int
    f_softargmax_fn: callable
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # Actor
        actor = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor = activation(actor)
        actor = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor)
        actor = activation(actor)
        logits_raw = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor)

        # Tsallis softmax parametrisation
        probs = self.f_softargmax_fn(logits_raw)  # shape (..., A), sums to 1
        # To use distrax.Categorical, we pass log-probs then softmax(log p) = p
        logits = jnp.log(probs + 1e-20)
        pi = distrax.Categorical(logits=logits)

        # Critic
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray       # original env reward
    reg_reward: jnp.ndarray   # Tsallis-regularised reward
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: Any


def make_train(config):
    # Compute number of updates and minibatch size
    config["TOTAL_TIMESTEPS"] = int(config["TOTAL_TIMESTEPS"])
    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    print(f"num updates: {config['NUM_UPDATES']}")
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    # Create environment
    env, env_params = gymnax.make(config["ENV_NAME"], size=config["ENV_SIZE"])
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    def linear_schedule(base_lr, count):
        config["LR_END"] = config.get("LR_END", 0.0)
        # Decay over NUM_UPDATES (in units of epochs * minibatches)
        updates_per_epoch = config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]
        frac = 1.0 - (count // updates_per_epoch) / config["NUM_UPDATES"]
        frac = jnp.clip(frac, 0.0, 1.0)
        return base_lr * frac + (1.0 - frac) * config["LR_END"]

    def train(entropy_coeff, reg_alpha, param_alpha, learning_rate, rng):
        """
        entropy_coeff: used here as β, the regularisation strength
        reg_alpha: α for the regularisation divergence
        param_alpha: α for the Tsallis-softmax parameterisation
        """
        # INIT divergences
        param_divergence = AlphaDivergence(alpha=param_alpha, use_implicit_diff=True)
        reg_divergence = AlphaDivergence(alpha=reg_alpha, use_implicit_diff=True)

        beta = entropy_coeff  # rename conceptually

        # Tsallis softargmax parametrisation
        def alpha_softargmax(scores):
            # scores: (..., A)
            flat_scores = scores.reshape(-1, scores.shape[-1])

            def _softarg(v):
                return param_divergence.softargmax(v, prior=None, beta=1.0)

            probs = jax.vmap(_softarg)(flat_scores)
            return probs.reshape(scores.shape)

        # INIT network
        network = ActorCritic(
            action_dim=env.action_space(env_params).n,
            activation=config["ACTIVATION"],
            f_softargmax_fn=alpha_softargmax,
        )
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)

        # Optimiser
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(
                    learning_rate=lambda count: linear_schedule(learning_rate, count),
                    eps=1e-5,
                ),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate, eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT env state
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # One full PPO update step: collect rollout + update
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)

                # Tsallis regularisation term: D_α(π(·|s)||uniform)
                probs = pi.probs  # (NUM_ENVS, A)
                num_actions = probs.shape[-1]
                uniform = jnp.ones_like(probs) / num_actions
                reg_term = reg_divergence.divergence(probs, uniform)  # (NUM_ENVS,)
                reg_reward = reward - beta * reg_term

                transition = Transition(
                    done=done,
                    action=action,
                    value=value,
                    reward=reward,
                    reg_reward=reg_reward,
                    log_prob=log_prob,
                    obs=last_obs,
                    info=info,
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE (on regularised reward)
            train_state, env_state, last_obs, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition: Transition):
                    gae, next_value = gae_and_next_value
                    done = transition.done
                    value = transition.value
                    reg_reward = transition.reg_reward
                    delta = reg_reward + config["GAMMA"] * next_value * (1.0 - done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1.0 - done) * gae
                    return (gae, value), gae

                (final_gae, _), advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                # Regularised value targets
                targets = advantages + traj_batch.value
                return advantages, targets

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minibatch(train_state, batch_info):
                    traj_batch_mb, advantages_mb, targets_mb = batch_info

                    def _loss_fn(params, traj_batch_mb, gae_mb, targets_mb):
                        # RERUN NETWORK
                            # obs: (B, ...)
                        pi, value = network.apply(params, traj_batch_mb.obs)
                        log_prob = pi.log_prob(traj_batch_mb.action)

                        # VALUE LOSS (clipped)
                        value_pred_clipped = traj_batch_mb.value + (
                            value - traj_batch_mb.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets_mb)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets_mb)
                        value_loss = 0.5 * jnp.maximum(
                            value_losses, value_losses_clipped
                        ).mean()

                        # ACTOR LOSS (PPO surrogate, using regularised advantages)
                        ratio = jnp.exp(log_prob - traj_batch_mb.log_prob)
                        gae_mb = (gae_mb - gae_mb.mean()) / (gae_mb.std() + 1e-8)
                        loss_actor1 = ratio * gae_mb
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae_mb
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

                        total_loss = loss_actor + config["VF_COEF"] * value_loss
                        return total_loss, (value_loss, loss_actor)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (total_loss, aux), grads = grad_fn(
                        train_state.params, traj_batch_mb, advantages_mb, targets_mb
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch_ep, advantages_ep, targets_ep, rng = update_state
                rng, _rng = jax.random.split(rng)

                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"

                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch_ep, advantages_ep, targets_ep)

                # Flatten time/env dims
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )

                # Shuffle
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )

                # Split into minibatches
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minibatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch_ep, advantages_ep, targets_ep, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            rng = update_state[-1]

            metric = traj_batch.info  # env-specific metrics (e.g. episode returns)
            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"], unroll=False
        )
        return {"runner_state": runner_state, "metrics": metric}

    # JIT with reg_alpha and param_alpha static
    return jax.jit(train, static_argnums=(1, 2))


def run():
    general_config = {
        "NUM_ENVS": 16,
        "NUM_STEPS": 32,
        "TOTAL_TIMESTEPS": 1_000_000,  # int, not float
        "UPDATE_EPOCHS": 16,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": True,
        "ENV_NAME": "DeepSea-bsuite",
        "ENV_SIZE": 13,
    }

    rng = jax.random.PRNGKey(42)
    train_fn = make_train(general_config)
    n_seeds = 25

    lrs = jnp.array([1e-4, 3e-4, 1e-3])
    entropy_coeff = jnp.array([0.001, 0.01, 0.1, 1.0])  # β values
    reg_alpha = [0.1, 0.3, 0.5, 0.7, 0.9]
    param_alpha = [0.1, 0.3, 0.5, 0.7, 0.9]
    seeds = jax.random.split(rng, n_seeds)

    # vmap over seeds, LR, entropy_coeff (β)
    train_vmap_seeds = jax.vmap(
        train_fn, in_axes=(None, None, None, None, 0)
    )  # seeds
    train_vmap_lrs = jax.vmap(
        train_vmap_seeds, in_axes=(None, None, None, 0, None)
    )  # lrs
    train_vmap_entropy = jax.vmap(
        train_vmap_lrs, in_axes=(0, None, None, None, None)
    )  # entropy_coeff

    entropy_values = np.array(entropy_coeff)
    lr_values = np.array(lrs)
    seed_values = np.array(seeds)

    records = []
    total_alpha_pairs = len(reg_alpha) * len(param_alpha)
    alpha_pairs = product(reg_alpha, param_alpha)

    for reg_value, param_value in tqdm(
        alpha_pairs, total=total_alpha_pairs, desc="alpha-grid"
    ):
        batch_results = jax.block_until_ready(
            train_vmap_entropy(entropy_coeff, reg_value, param_value, lrs, seeds)
        )
        print("Completed alpha pair:", reg_value, param_value)
        host_results = jax.device_get(batch_results)
        rewards = host_results["metrics"]["returned_episode_returns"]

        for e_idx, entropy_value in enumerate(entropy_values):
            for lr_idx, lr_value in enumerate(lr_values):
                for seed_idx, seed_value in enumerate(seed_values):
                    current_rewards = rewards[e_idx, lr_idx, seed_idx].mean(-1).reshape(-1)
                    record = {
                        "entropy_coeff": float(entropy_value),
                        "reg_alpha": float(reg_value),
                        "param_alpha": float(param_value),
                        "learning_rate": float(lr_value),
                        "seed": np.array(seed_value),
                        "rewards": np.array(current_rewards),
                    }
                    records.append(record)

    with open("deepsea_results_13.pkl", "wb") as fp:
        pickle.dump(records, fp)

    return records


if __name__ == "__main__":
    run()
