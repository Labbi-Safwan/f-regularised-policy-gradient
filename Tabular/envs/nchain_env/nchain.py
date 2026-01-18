
from typing import Any, Optional, Dict, Tuple

import jax
import jax.numpy as jnp
from flax import struct

from gymnax.environments import environment, spaces


@struct.dataclass
class EnvState(environment.EnvState):
    state: int     
    time: int        


@struct.dataclass
class EnvParams(environment.EnvParams):
    n: int = 10
    small_reward: float = 0.01
    big_reward: float = 1.0
    max_steps: int = 18   


class NChain(environment.Environment):

    def __init__(self, n: int = 10):
        super().__init__()
        self.n = n        # chain length

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(n=self.n, max_steps=self.n + 8)


    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: Any,
        params: EnvParams,
    ) -> Tuple[jax.Array, EnvState, jax.Array, jax.Array, Dict[str, Any]]:

        del key  

        s = state.state
        a = jnp.asarray(action, dtype=jnp.int32)

        rew_big = jnp.logical_and(s == params.n - 1, a == 1)
        rew_small = jnp.logical_and(s == 0, a == 0)
        reward = (
            rew_big.astype(jnp.float32) * params.big_reward
            + rew_small.astype(jnp.float32) * params.small_reward
        )


        s_right = jnp.where(s != params.n - 1, s + 1, s)
        s_left = jnp.where(s != 0, s - 1, s)
        s_next = jnp.where(a == 1, s_right, s_left)

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

    def get_obs(
        self,
        state: EnvState,
        params: EnvParams,
        key: Any = None,
    ) -> jax.Array:

        del key, params  
        idx = jnp.arange(self.n, dtype=jnp.int32)
        obs = (idx <= state.state).astype(jnp.float32)
        return obs

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        return (state.time >= params.max_steps).astype(jnp.bool_)

    def discount(self, state: EnvState, params: EnvParams) -> jax.Array:
        return 1.0 - self.is_terminal(state, params).astype(jnp.float32)


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
