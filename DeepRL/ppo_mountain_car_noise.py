# ppo_noisy_mountaincar_onefile_fixed.py

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import pickle

from flax.linen.initializers import constant, orthogonal
from typing import NamedTuple, Any, Tuple
from flax.training.train_state import TrainState
import distrax

import chex
from gymnax.environments import environment, spaces
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper

from itertools import product
from tqdm import tqdm

from fdiv import AlphaDivergence


# ============================================================
# 1) JAX MountainCar (bsuite-like)
# ============================================================

class MountainCarParams(NamedTuple):
    min_pos: float = -1.2
    max_pos: float = 0.6
    max_speed: float = 0.07
    goal_pos: float = 0.5
    force: float = 0.001
    gravity: float = 0.0025
    max_steps: int = 1000


class MountainCarState(NamedTuple):
    position: jnp.ndarray
    velocity: jnp.ndarray
    t: jnp.ndarray


class MountainCarEnv(environment.Environment):
    """
    bsuite-like MountainCar:
      - reset position ~ Uniform[-0.6, -0.4], velocity=0, t=0
      - actions {0,1,2} -> (action-1) in {-1,0,+1}
      - reward = -1 each step
      - done if position >= goal_pos OR t >= max_steps
      - obs = [position, velocity, t/max_steps] (shape (3,))
    """
    def __init__(self):
        super().__init__()

    @property
    def default_params(self) -> MountainCarParams:
        return MountainCarParams()

    def observation_space(self, params: MountainCarParams):
        high = jnp.array([jnp.finfo(jnp.float32).max] * 3)
        return spaces.Box(-high, high, (3,), jnp.float32)

    def action_space(self, params: MountainCarParams):
        return spaces.Discrete(3)

    def reset(self, key: chex.PRNGKey, params: MountainCarParams) -> Tuple[jnp.ndarray, MountainCarState]:
        position = jax.random.uniform(key, (), minval=-0.6, maxval=-0.4)
        velocity = jnp.array(0.0, dtype=jnp.float32)
        t = jnp.array(0, dtype=jnp.int32)

        state = MountainCarState(position=position, velocity=velocity, t=t)

        # ---- FIX: do NOT use float(params.max_steps) under jit ----
        max_steps_f = jnp.asarray(params.max_steps, dtype=jnp.float32)
        obs = jnp.array([position, velocity, t.astype(jnp.float32) / max_steps_f], dtype=jnp.float32)

        return obs, state

    def step(
        self,
        key: chex.PRNGKey,
        state: MountainCarState,
        action: jnp.ndarray,
        params: MountainCarParams,
    ) -> Tuple[jnp.ndarray, MountainCarState, jnp.ndarray, jnp.ndarray, dict]:

        # time
        t = state.t + jnp.array(1, dtype=jnp.int32)

        # reward = -1
        reward = jnp.array(-1.0, dtype=jnp.float32)

        # dynamics
        position = state.position
        velocity = state.velocity

        velocity = velocity + (action.astype(jnp.float32) - 1.0) * params.force + jnp.cos(3.0 * position) * (-params.gravity)
        velocity = jnp.clip(velocity, -params.max_speed, params.max_speed)

        position = position + velocity
        position = jnp.clip(position, params.min_pos, params.max_pos)

        velocity = jnp.where(position == params.min_pos, jnp.clip(velocity, 0.0, params.max_speed), velocity)

        next_state = MountainCarState(position=position, velocity=velocity, t=t)

        # ---- FIX: do NOT use float(params.max_steps) under jit ----
        max_steps_f = jnp.asarray(params.max_steps, dtype=jnp.float32)
        obs = jnp.array([position, velocity, t.astype(jnp.float32) / max_steps_f], dtype=jnp.float32)

        done = (position >= params.goal_pos) | (t >= params.max_steps)

        info = {"position": position, "velocity": velocity, "t": t}
        return obs, next_state, reward, done.astype(jnp.float32), info


# ============================================================
# 2) Reward noise wrapper (JAX, bsuite-style)
#     IMPORTANT FIX: step signature must be (key, state, action, params)
# ============================================================

class RewardNoiseState(NamedTuple):
    env_state: Any
    is_first: jnp.ndarray  # bool scalar


class RewardNoiseWrapperJAX:
    """
    Adds reward noise: r <- r + sigma * N(0,1)
    BUT does NOT add noise on the first step after reset (bsuite-style).

    NOTE: noise_scale is stored in the wrapper instance so that
    step signature remains gymnax-compatible:
      step(key, state, action, params)
    """
    def __init__(self, env, noise_scale: float):
        self._env = env
        # store as JAX scalar (safe inside jit)
        self.noise_scale = jnp.asarray(noise_scale, dtype=jnp.float32)

    def observation_space(self, params):
        return self._env.observation_space(params)

    def action_space(self, params):
        return self._env.action_space(params)

    def reset(self, key, params):
        obs, st = self._env.reset(key, params)
        return obs, RewardNoiseState(env_state=st, is_first=jnp.array(True))

    def step(self, key, state: RewardNoiseState, action, params):
        key_env, key_noise = jax.random.split(key, 2)
        obs, st, reward, done, info = self._env.step(key_env, state.env_state, action, params)

        noise = self.noise_scale * jax.random.normal(key_noise, shape=reward.shape)
        reward_noisy = jnp.where(state.is_first, reward, reward + noise)

        next_state = RewardNoiseState(env_state=st, is_first=jnp.array(False))
        return obs, next_state, reward_noisy, done, info

    @property
    def raw_env(self):
        wrapped = self._env
        if hasattr(wrapped, "raw_env"):
            return wrapped.raw_env
        return wrapped


# ============================================================
# 3) PPO network + transition
# ============================================================

class ActorCritic(nn.Module):
    action_dim: int
    f_softargmax_fn: callable
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh

        # Actor
        actor = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        actor = activation(actor)
        actor = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor)
        actor = activation(actor)
        logits_raw = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor)

        probs = self.f_softargmax_fn(logits_raw)
        logits = jnp.log(probs + 1e-20)
        pi = distrax.Categorical(logits=logits)

        # Critic
        critic = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        critic = activation(critic)
        critic = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    reg_reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: Any


# ============================================================
# 4) Train function
# ============================================================

def make_train(config):
    config["TOTAL_TIMESTEPS"] = int(config["TOTAL_TIMESTEPS"])
    config["NUM_UPDATES"] = int(config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])
    print(f"num updates: {config['NUM_UPDATES']}")
    config["MINIBATCH_SIZE"] = config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]

    # ----- env -----
    base_env = MountainCarEnv()
    env_params = base_env.default_params

    # noisy rewards (noise scale stored inside wrapper instance)
    noisy_env = RewardNoiseWrapperJAX(base_env, noise_scale=float(config["NOISE_SCALE"]))

    # gymnax wrappers (now compatible because noisy_env.step has 4-arg signature)
    noisy_env = FlattenObservationWrapper(noisy_env)
    noisy_env = LogWrapper(noisy_env)

    def linear_schedule(base_lr, count):
        config["LR_END"] = config.get("LR_END", 0.0)
        updates_per_epoch = config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]
        frac = 1.0 - (count // updates_per_epoch) / config["NUM_UPDATES"]
        frac = jnp.clip(frac, 0.0, 1.0)
        return base_lr * frac + (1.0 - frac) * config["LR_END"]

    def train(entropy_coeff, reg_alpha, param_alpha, learning_rate, rng):
        param_divergence = AlphaDivergence(alpha=param_alpha, use_implicit_diff=True)
        reg_divergence = AlphaDivergence(alpha=reg_alpha, use_implicit_diff=True)
        beta = entropy_coeff

        def alpha_softargmax(scores):
            flat_scores = scores.reshape(-1, scores.shape[-1])

            def _softarg(v):
                return param_divergence.softargmax(v, prior=None, beta=1.0)

            probs = jax.vmap(_softarg)(flat_scores)
            return probs.reshape(scores.shape)

        network = ActorCritic(
            action_dim=noisy_env.action_space(env_params).n,
            activation=config["ACTIVATION"],
            f_softargmax_fn=alpha_softargmax,
        )

        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(noisy_env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=lambda count: linear_schedule(learning_rate, count), eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate, eps=1e-5),
            )

        train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

        # INIT env state
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(noisy_env.reset, in_axes=(0, None))(reset_rng, env_params)

        def _update_step(runner_state, unused):
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                # IMPORTANT: now step is 4-arg, wrapper-safe
                obsv, env_state, reward, done, info = jax.vmap(
                    noisy_env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)

                probs = pi.probs
                num_actions = probs.shape[-1]
                uniform = jnp.ones_like(probs) / num_actions
                reg_term = reg_divergence.divergence(probs, uniform)
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
                return (train_state, env_state, obsv, rng), transition

            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["NUM_STEPS"])

            train_state, env_state, last_obs, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_adv(gae_and_next_value, transition: Transition):
                    gae, next_value = gae_and_next_value
                    delta = transition.reg_reward + config["GAMMA"] * next_value * (1.0 - transition.done) - transition.value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1.0 - transition.done) * gae
                    return (gae, transition.value), gae

                (_, _), advantages = jax.lax.scan(
                    _get_adv,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                targets = advantages + traj_batch.value
                return advantages, targets

            advantages, targets = _calculate_gae(traj_batch, last_val)

            def _update_epoch(update_state, unused):
                def _update_minibatch(train_state, batch_info):
                    traj_mb, adv_mb, tgt_mb = batch_info

                    def _loss_fn(params, traj_mb, adv_mb, tgt_mb):
                        pi, value = network.apply(params, traj_mb.obs)
                        log_prob = pi.log_prob(traj_mb.action)

                        value_pred_clipped = traj_mb.value + (value - traj_mb.value).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"]
                        )
                        value_losses = jnp.square(value - tgt_mb)
                        value_losses_clipped = jnp.square(value_pred_clipped - tgt_mb)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        ratio = jnp.exp(log_prob - traj_mb.log_prob)
                        adv_mb = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)
                        loss_actor1 = ratio * adv_mb
                        loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * adv_mb
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

                        total_loss = loss_actor + config["VF_COEF"] * value_loss
                        return total_loss, (value_loss, loss_actor)

                    (total_loss, _aux), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
                        train_state.params, traj_mb, adv_mb, tgt_mb
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_ep, adv_ep, tgt_ep, rng = update_state
                rng, _rng = jax.random.split(rng)

                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]

                perm = jax.random.permutation(_rng, batch_size)
                batch = (traj_ep, adv_ep, tgt_ep)
                batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
                batch = jax.tree_util.tree_map(lambda x: jnp.take(x, perm, axis=0), batch)

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])),
                    batch,
                )

                train_state, total_loss = jax.lax.scan(_update_minibatch, train_state, minibatches)
                return (train_state, traj_ep, adv_ep, tgt_ep, rng), total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, _ = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
            train_state = update_state[0]
            rng = update_state[-1]

            metric = traj_batch.info
            return (train_state, env_state, last_obs, rng), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"], unroll=False)
        return {"runner_state": runner_state, "metrics": metric}

    return jax.jit(train, static_argnums=(1, 2))


# ============================================================
# 5) Run gridsearch
# ============================================================

def run():
    general_config = {
        "NUM_ENVS": 16,
        "NUM_STEPS": 32,
        "TOTAL_TIMESTEPS": 1_000_000,
        "UPDATE_EPOCHS": 16,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": True,

        # noisy mountain car
        "NOISE_SCALE": 0.1,
    }

    rng = jax.random.PRNGKey(42)
    train_fn = make_train(general_config)
    n_seeds = 25

    lrs = jnp.array([1e-4, 3e-4, 1e-3])
    entropy_coeff = jnp.array([0.001, 0.01, 0.1, 1.0])
    reg_alpha = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    param_alpha = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    seeds = jax.random.split(rng, n_seeds)

    train_vmap_seeds = jax.vmap(train_fn, in_axes=(None, None, None, None, 0))
    train_vmap_lrs = jax.vmap(train_vmap_seeds, in_axes=(None, None, None, 0, None))
    train_vmap_entropy = jax.vmap(train_vmap_lrs, in_axes=(0, None, None, None, None))

    entropy_values = np.array(entropy_coeff)
    lr_values = np.array(lrs)
    seed_values = np.array(seeds)

    records = []
    alpha_pairs = product(reg_alpha, param_alpha)

    for reg_value, param_value in tqdm(alpha_pairs, total=len(reg_alpha) * len(param_alpha), desc="alpha-grid"):
        batch_results = jax.block_until_ready(
            train_vmap_entropy(entropy_coeff, reg_value, param_value, lrs, seeds)
        )
        host_results = jax.device_get(batch_results)
        rewards = host_results["metrics"]["returned_episode_returns"]

        for e_idx, entropy_value in enumerate(entropy_values):
            for lr_idx, lr_value in enumerate(lr_values):
                for seed_idx, seed_value in enumerate(seed_values):
                    current_rewards = rewards[e_idx, lr_idx, seed_idx].mean(-1).reshape(-1)
                    records.append({
                        "noise_scale": float(general_config["NOISE_SCALE"]),
                        "entropy_coeff": float(entropy_value),
                        "reg_alpha": float(reg_value),
                        "param_alpha": float(param_value),
                        "learning_rate": float(lr_value),
                        "seed": np.array(seed_value),
                        "rewards": np.array(current_rewards),
                    })

    out = "mountaincar_noisy_results.pkl"
    with open(out, "wb") as fp:
        pickle.dump(records, fp)
    print("Saved:", out)
    return records


if __name__ == "__main__":
    run()
