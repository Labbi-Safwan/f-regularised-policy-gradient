# envs/deepsea_stochastic_env/deepsea_stochastic_jax.py

from typing import Any, Optional, Dict, Tuple

import jax
import jax.numpy as jnp
from flax import struct

from gymnax.environments import environment, spaces


# ---------------------------------------------------------------------
#  State and Params
# ---------------------------------------------------------------------
@struct.dataclass
class EnvState(environment.EnvState):
    row: int
    column: int
    bad_episode: bool
    total_bad_episodes: int
    denoised_return: float
    optimal_return: float
    action_mapping: jax.Array
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    # stochastic by default (deterministic=False)
    deterministic: bool = False
    sample_action_map: bool = False
    unscaled_move_cost: float = 0.01
    randomize_actions: bool = True
    max_steps_in_episode: int = 2000


# ---------------------------------------------------------------------
#  Stochastic DeepSea env (JAX)
# ---------------------------------------------------------------------
class DeepSeaDet(environment.Environment):
    """
    JAX implementation of bsuite DeepSea with stochastic dynamics.

    Grid of size N x N.
    - Tabular states: all grid cells plus an implicit terminal condition handled
      by (row >= N) in get_obs / is_terminal.
    - Actions: 0/1, interpreted via a per-state action_mapping:
        action_right = (a == action_mapping[row, col])
    - Dynamics and rewards are exactly as in bsuite / your tabular
      StochasticDeepSea:
        * slip probability 1/N when going 'right' (if deterministic=False)
        * Gaussian noise at bottom row edges when stochastic
        * move cost unscaled_move_cost / N when right_cond holds
        * +1 reward for going right in the last column.
    """

    def __init__(self, size: int = 8):
        super().__init__()
        self.size = size
        # default action mapping (used when sample_action_map=False and
        # randomize_actions=False, or carried across episodes)
        self._default_action_mapping = jnp.ones((size, size), dtype=jnp.int32)

    # ------------------------------------------------------------------
    # Default parameters
    # ------------------------------------------------------------------
    @property
    def default_params(self) -> EnvParams:
        # stochastic: deterministic=False by default
        return EnvParams(
            deterministic=True,
            sample_action_map=False,
            unscaled_move_cost=0.01,
            randomize_actions=True,
            max_steps_in_episode=2000,
        )

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------
    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: Any,               # <- was int | float | jax.Array
        params: EnvParams,
    ) -> Tuple[jax.Array, EnvState, jax.Array, jax.Array, Dict[str, Any]]:
        """Perform single timestep state transition (stochastic)."""

        key_reward, key_trans = jax.random.split(key)
        rand_reward = jax.random.normal(key_reward, shape=())
        # slip variable: True with prob (1 - 1/N)
        rand_trans_cond = (
            jax.random.uniform(key_trans, shape=(), minval=0.0, maxval=1.0)
            > 1.0 / self.size
        )

        a = jnp.asarray(action, dtype=jnp.int32)
        action_right = a == state.action_mapping[state.row, state.column]
        right_rand_cond = jnp.logical_or(rand_trans_cond, params.deterministic)
        right_cond = jnp.logical_and(action_right, right_rand_cond)

        # reward + denoised_return update
        reward, denoised_return = _step_reward(
            state, action_right, right_cond, rand_reward, self.size, params
        )
        # transition update
        column, row, bad_episode = _step_transition(
            state, action_right, right_cond, self.size
        )

        new_state = state.replace(
            row=row,
            column=column,
            bad_episode=bad_episode,
            denoised_return=denoised_return,
            time=state.time + 1,
        )

        done = self.is_terminal(new_state, params)
        new_state = new_state.replace(
            total_bad_episodes=new_state.total_bad_episodes
            + done * new_state.bad_episode
        )

        obs = jax.lax.stop_gradient(self.get_obs(new_state, params))
        new_state = jax.lax.stop_gradient(new_state)
        info: Dict[str, Any] = {"discount": self.discount(new_state, params)}

        return obs, new_state, reward, done, info

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> Tuple[jax.Array, EnvState]:
        """Reset environment state by sampling initial position & mapping."""

        # optimal_return used only for logging; same as bsuite:
        optimal_no_cost = (1 - params.deterministic) * (1 - 1 / self.size) ** (
            self.size - 1
        ) + params.deterministic * 1.0
        optimal_return = optimal_no_cost - params.unscaled_move_cost

        # draw new action map or reuse previous
        a_map_rand = jax.random.bernoulli(key, 0.5, (self.size, self.size)).astype(
            jnp.int32
        )
        a_map_determ = self._default_action_mapping

        new_a_map_cond = jnp.logical_and(
            jnp.logical_not(params.deterministic), params.sample_action_map
        )
        old_a_map_cond = jnp.logical_and(
            jnp.logical_not(params.deterministic),
            jnp.logical_not(params.sample_action_map),
        )
        action_mapping = (
            params.deterministic * a_map_determ
            + new_a_map_cond * a_map_rand
            + old_a_map_cond * self._default_action_mapping
        )

        state = EnvState(
            row=0,
            column=0,
            bad_episode=False,
            total_bad_episodes=0,
            denoised_return=0.0,
            optimal_return=optimal_return,
            action_mapping=action_mapping,
            time=0,
        )

        obs = self.get_obs(state, params)
        return obs, state

    # ------------------------------------------------------------------
    # Observation / termination / discount
    # ------------------------------------------------------------------
    def get_obs(
        self,
        state: EnvState,
        params: Optional[EnvParams] = None,   # <- was EnvParams | None
        key: Any = None,
    ) -> jax.Array:
        """
        One-hot position on N x N grid until row >= N, then all zeros
        (absorbing).
        """
        del key  # unused, for gymnax API compatibility
        obs_end = jnp.zeros((self.size, self.size), dtype=jnp.float32)
        end_cond = state.row >= self.size
        obs_upd = obs_end.at[state.row, state.column].set(1.0)
        return jax.lax.select(end_cond, obs_end, obs_upd)

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Terminal if we reached bottom row or max steps."""
        done_row = state.row == self.size
        done_steps = state.time >= params.max_steps_in_episode
        return jnp.logical_or(done_row, done_steps)

    def discount(self, state: EnvState, params: EnvParams) -> jax.Array:
        return 1.0 - self.is_terminal(state, params).astype(jnp.float32)

    # ------------------------------------------------------------------
    # Meta info / spaces
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return "StochasticDeepSea-bsuite"

    @property
    def num_actions(self) -> int:
        return 2

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        return spaces.Discrete(2)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(0.0, 1.0, (self.size, self.size), jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        return spaces.Dict(
            {
                "row": spaces.Discrete(self.size),
                "column": spaces.Discrete(self.size),
                "bad_episode": spaces.Discrete(2),
                "total_bad_episodes": spaces.Discrete(2000),
                "denoised_return": spaces.Box(-1e6, 1e6, ()),
                "optimal_return": spaces.Box(-1e6, 1e6, ()),
                "action_mapping": spaces.Box(
                    0,
                    1,
                    (self.size, self.size),
                    dtype=jnp.int32,
                ),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )


# ---------------------------------------------------------------------
#  Reward / transition helpers (same as your original JAX DeepSea)
# ---------------------------------------------------------------------
def _step_reward(
    state: EnvState,
    action_right: jax.Array,
    right_cond: jax.Array,
    rand_reward: jax.Array,
    size: int,
    params: EnvParams,
) -> Tuple[jax.Array, jax.Array]:
    """Reward for the selected action, including move cost and noise."""
    reward = 0.0

    # +1 if at rightmost column and taking 'right'
    rew_cond = jnp.logical_and(state.column == size - 1, action_right)
    reward += rew_cond.astype(jnp.float32)
    denoised_return = state.denoised_return + rew_cond.astype(jnp.float32)

    # Gaussian noise at chain ends (bottom row & edge columns) when stochastic.
    col_at_edge = jnp.logical_or(state.column == 0, state.column == size - 1)
    chain_end = jnp.logical_and(state.row == size - 1, col_at_edge)
    det_chain_end = jnp.logical_and(chain_end, jnp.logical_not(params.deterministic))
    reward += rand_reward * det_chain_end.astype(jnp.float32)

    # Move cost for "right_cond" transitions
    reward -= right_cond.astype(jnp.float32) * params.unscaled_move_cost / size
    return reward, denoised_return


def _step_transition(
    state: EnvState,
    action_right: jax.Array,
    right_cond: jax.Array,
    size: int,
) -> Tuple[jax.Array, int, jax.Array]:
    """State transition for the selected action (bsuite-style)."""

    # Standard right path transition (with slip)
    column = jax.lax.select(
        right_cond,
        jnp.clip(state.column + 1, 0, size - 1),
        state.column,
    )

    # If action not right, move left
    column = jax.lax.select(
        action_right,
        column,
        jnp.clip(state.column - 1, 0, size - 1),
    )

    # Track whether we ever left the optimal right path (bad_episode)
    right_wrong_cond = jnp.logical_and(
        jnp.logical_not(action_right),
        state.row == column,
    )
    bad_episode = jax.lax.select(right_wrong_cond, True, state.bad_episode)

    row = state.row + 1
    return column, row, bad_episode

