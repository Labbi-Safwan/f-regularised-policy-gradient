# envs/nchain_env/nchain_jax.py

from typing import Any, Optional, Dict, Tuple

import jax
import jax.numpy as jnp
from flax import struct

from gymnax.environments import environment, spaces


@struct.dataclass
class EnvState(environment.EnvState):
    state: int       # chain index in {0, ..., n-1}
    time: int        # step counter


@struct.dataclass
class EnvParams(environment.EnvParams):
    n: int = 10
    small_reward: float = 0.01
    big_reward: float = 1.0
    max_steps: int = 18   # default: n + 8, but you can override


class NChain(environment.Environment):
    """
    JAX implementation of the tabular N-Chain environment.

    - States: {0, ..., n-1}
    - Start state: 1
    - Actions: 0 = left, 1 = right
    - Dynamics (deterministic):
        * if a == 1 (right):
            - if s != n-1: s' = s + 1
            - else:        s' = s
        * if a == 0 (left):
            - if s != 0:   s' = s - 1
            - else:        s' = s
    - Rewards:
        * if s == n-1 and a == 1: big_reward
        * elif s == 0 and a == 0: small_reward
        * else: 0
    - Termination:
        * done when time >= max_steps.
    - Observation:
        * vector of length n: 1{ i <= state }, i=0,...,n-1.
    """

    def __init__(self, n: int = 10):
        super().__init__()
        self.n = n        # chain length

    # ------------------------------------------------------------------
    # Default parameters
    # ------------------------------------------------------------------
    @property
    def default_params(self) -> EnvParams:
        return EnvParams(n=self.n, max_steps=self.n + 8)

    # ------------------------------------------------------------------
    # Core step & reset
    # ------------------------------------------------------------------
    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: Any,
        params: EnvParams,
    ) -> Tuple[jax.Array, EnvState, jax.Array, jax.Array, Dict[str, Any]]:

        del key  # not used (but kept for gymnax API compatibility)

        # current state
        s = state.state
        a = jnp.asarray(action, dtype=jnp.int32)

        # ---- Reward ----
        rew_big = jnp.logical_and(s == params.n - 1, a == 1)
        rew_small = jnp.logical_and(s == 0, a == 0)
        reward = (
            rew_big.astype(jnp.float32) * params.big_reward
            + rew_small.astype(jnp.float32) * params.small_reward
        )

        # ---- Transition ----
        # right move
        s_right = jnp.where(s != params.n - 1, s + 1, s)
        # left move
        s_left = jnp.where(s != 0, s - 1, s)
        # select based on action
        s_next = jnp.where(a == 1, s_right, s_left)

        # update time
        time_next = state.time + 1

        next_state = state.replace(state=s_next, time=time_next)

        done = self.is_terminal(next_state, params)
        info: Dict[str, Any] = {"discount": self.discount(next_state, params)}

        obs = jax.lax.stop_gradient(self.get_obs(next_state, params))
        next_state = jax.lax.stop_gradient(next_state)

        return obs, next_state, reward, done, info

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> Tuple[jax.Array, EnvState]:
        """Reset to start state s=1, time=0."""
        del key  # unused
        state = EnvState(state=1, time=0)
        obs = self.get_obs(state, params)
        return obs, state

    # ------------------------------------------------------------------
    # Observation, termination, discount
    # ------------------------------------------------------------------
    def get_obs(
        self,
        state: EnvState,
        params: EnvParams,
        key: Any = None,
    ) -> jax.Array:
        """
        Observation: vector v of length n with v[i] = 1 if i <= state, else 0,
        encoded as float32.
        """
        del key, params  # params is traced in jit; we only use static self.n
        idx = jnp.arange(self.n, dtype=jnp.int32)
        obs = (idx <= state.state).astype(jnp.float32)
        return obs

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Episode ends once time >= max_steps."""
        return (state.time >= params.max_steps).astype(jnp.bool_)

    def discount(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Standard discount: 0 if terminal, 1 otherwise."""
        return 1.0 - self.is_terminal(state, params).astype(jnp.float32)

    # ------------------------------------------------------------------
    # Meta info / spaces
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return "NChain-jax"

    @property
    def num_actions(self) -> int:
        return 2

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        return spaces.Discrete(2)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(0.0, 1.0, (params.n,), jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        return spaces.Dict(
            {
                "state": spaces.Discrete(params.n),
                "time": spaces.Discrete(params.max_steps + 1),
            }
        )
