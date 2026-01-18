# algos_jax.py

from functools import partial
from typing import Callable, Dict, Tuple

import chex
import jax
import jax.numpy as jnp

from fdiv import AlphaDivergence


# ---------------------------------------------------------------------
#  Parametrisations: π_θ(s,·) and log π_θ(s,a)
# ---------------------------------------------------------------------
def softmax_policy_theta_row(theta_row: chex.Array) -> chex.Array:
    """Standard softmax over a single row θ_s."""
    logits = theta_row - jnp.max(theta_row)
    exp = jnp.exp(logits)
    return exp / jnp.sum(exp)


def escort_policy_theta_row(theta_row: chex.Array, p: float) -> chex.Array:
    """Escort transform: π(a) ∝ |θ(a)|^p."""
    z = jnp.abs(theta_row) ** p
    z_sum = jnp.sum(z)
    probs = jax.lax.cond(
        z_sum <= 1e-12,
        lambda: jnp.ones_like(z) / z.shape[0],
        lambda: z / z_sum,
    )
    return probs


def hadamard_policy_theta_row(theta_row: chex.Array) -> chex.Array:
    """Hadamard parametrisation: π(a) ∝ θ(a)^2."""
    z = theta_row ** 2
    z_sum = jnp.sum(z)
    probs = jax.lax.cond(
        z_sum <= 1e-12,
        lambda: jnp.ones_like(z) / z.shape[0],
        lambda: z / z_sum,
    )
    return probs


def make_logpi_fn(
    policy_row_fn: Callable[[chex.Array], chex.Array],
):
    """
    Build log π_θ(a|s) as a scalar function θ -> log π_θ(a|s),
    so we can use jax.grad on it.

    θ has shape (S, A), state ∈ {0..S-1}, action ∈ {0..A-1}.
    """

    def log_pi(theta: chex.Array, state: int, action: int) -> chex.Array:
        theta_s = theta[state]  # shape (A,)
        probs_s = policy_row_fn(theta_s)
        probs_s = jnp.clip(probs_s, 1e-12, jnp.inf)
        probs_s = probs_s / jnp.sum(probs_s)
        return jnp.log(probs_s[action])

    grad_log_pi = jax.grad(log_pi)
    return log_pi, grad_log_pi


# ---------------------------------------------------------------------
#  Helpers for f-regularisation (Tsallis / KL)
# ---------------------------------------------------------------------
def _f_divergence_tsallis(
    probs: chex.Array,
    prior: chex.Array,
    alpha: float,
) -> chex.Array:
    """
    Tsallis / alpha-divergence D_f(probs || prior) with the same formula
    as your numpy code.

    - alpha = 1   → KL divergence.
    - alpha != 1  → Tsallis-like f-divergence.
    """
    u = probs / prior
    if abs(alpha - 1.0) < 1e-8:
        # KL(probs || prior)
        log_ratio = jnp.log(probs / prior)
        return jnp.sum(probs * log_ratio)
    else:
        # (u^α - α u + α - 1)/(α(α-1))
        f_u = (u ** alpha - alpha * u + alpha - 1.0) / (alpha * (alpha - 1.0))
        return jnp.sum(prior * f_u)


def _f_vector_F(
    probs: chex.Array,
    prior: chex.Array,
    alpha: float,
    S: int,
    A: int,
    state: int,
) -> chex.Array:
    """
    Vector F(s,·) in your numpy code, embedded in an (S,A) array.

      For α < 1:
        f''(u) = ref * u^{2-α}, W = Σ f'', w = f''/W,
        f'(u) = (u^{α-1} - 1)/(α-1)

      For α >= 1:
        f''(u) = ref * u, W = Σ f'', w = f''/W,
        f'(u) = log u
    """
    ref_probs = prior
    u = probs / ref_probs

    def branch_alpha_lt1():
        fpp = ref_probs * (u ** (2.0 - alpha))
        W = jnp.sum(fpp)
        w = fpp / W
        f_u_prime = (u ** (alpha - 1.0) - 1.0) / (alpha - 1.0)
        divergence_f_prime = jnp.sum(w * f_u_prime)
        F_row = W * w * (f_u_prime - divergence_f_prime)
        return F_row

    def branch_alpha_ge1():
        fpp = ref_probs * u
        W = jnp.sum(fpp)
        w = fpp / W
        f_u_prime = jnp.log(u)
        divergence_f_prime = jnp.sum(w * f_u_prime)
        F_row = W * w * (f_u_prime - divergence_f_prime)
        return F_row

    if alpha < 1.0:
        F_row = branch_alpha_lt1()
    else:
        F_row = branch_alpha_ge1()

    F = jnp.zeros((S, A), dtype=probs.dtype)
    F = F.at[state].set(F_row)
    return F


# ---------------------------------------------------------------------
#  Single REINFORCE update (model-free, fully JAX)
# ---------------------------------------------------------------------
def pg_update_step(
    key: chex.PRNGKey,
    theta: chex.Array,                # (S, A)
    env,
    params,
    gamma: float,
    H: int,
    B: int,
    policy_row_fn: Callable[[chex.Array], chex.Array],
    grad_log_pi_fn: Callable[[chex.Array, int, int], chex.Array],
    step_size: float = 1e-2,
    logbarrier_lambda: float = 0.0,
    # f-regularisation (used only for fPG)
    use_f_reg: bool = False,
    prior: chex.Array | None = None,
    alpha: float = 1.0,
    f_temp: float = 0.0,
    # Riemannian step on sphere (used only for Hadamard)
    riemannian: bool = False,
) -> Tuple[chex.PRNGKey, chex.Array, float]:
    """
    One REINFORCE update step (Monte Carlo, model-free):

        θ_{k+1} = θ_k + η * g(θ_k)

    Where g(θ_k) is:
      - vanilla REINFORCE gradient for escort/logbarrier;
      - REINFORCE + f-regularisation (divergence + F vector) for fPG;
      - Riemannian gradient on the sphere for Hadamard (if riemannian=True).

    JAX version:
      - Scan over horizon H for each episode.
      - vmap episodes over batch B.
    """
    S, A = theta.shape

    # ---------- One episode (no batching) ----------
    def run_one_episode(key_ep: chex.PRNGKey) -> Tuple[chex.Array, chex.Array]:
        # Reset environment
        key_reset, key_rollout = jax.random.split(key_ep)
        obs0, env_state0 = env.reset(key_reset, params)
        done0 = jnp.array(False)

        def step_fn(carry, _):
            key_c, env_state, obs, done = carry

            # Alive mask = 1 if episode not done yet at this step
            alive = 1.0 - done.astype(jnp.float32)

            # State index from one-hot observation
            s_idx = jnp.argmax(jnp.ravel(obs)).astype(jnp.int32)

            theta_s = theta[s_idx]
            probs_s = policy_row_fn(theta_s)
            probs_s = jnp.clip(probs_s, 1e-12, jnp.inf)
            probs_s = probs_s / jnp.sum(probs_s)

            key_c, key_a, key_step = jax.random.split(key_c, 3)
            a = jax.random.categorical(key_a, jnp.log(probs_s)).astype(jnp.int32)

            def do_step(args):
                key_step_inner, env_state_inner, a_inner = args
                obs_next, env_state_next, reward, done_env, _ = env.step(
                    key_step_inner, env_state_inner, a_inner, params
                )
                return obs_next, env_state_next, reward, done_env

            def do_nothing(args):
                key_step_inner, env_state_inner, a_inner = args
                del key_step_inner, a_inner
                return (
                    obs,
                    env_state_inner,
                    jnp.array(0.0, dtype=jnp.float32),
                    jnp.array(True),
                )

            obs_next, env_state_next, reward, done_env = jax.lax.cond(
                done,
                do_nothing,
                do_step,
                operand=(key_step, env_state, a),
            )

            done_next = jnp.logical_or(done, done_env)

            # We also store probs_s for f-regularisation
            step_out = (s_idx, a, reward, alive, probs_s)
            carry_next = (key_c, env_state_next, obs_next, done_next)
            return carry_next, step_out

        carry0 = (key_rollout, env_state0, obs0, done0)
        (_, _, _, _), (states_idx, actions, rewards, alive, probs_t) = jax.lax.scan(
            step_fn,
            carry0,
            xs=None,
            length=H,
        )

        # Discounted returns G_t (backwards scan)
        def discount_step(carry, r_t):
            G_next = carry
            G_t = r_t + gamma * G_next
            return G_t, G_t

        _, G_rev = jax.lax.scan(
            discount_step,
            jnp.array(0.0, dtype=jnp.float32),
            rewards[::-1],
        )
        G = G_rev[::-1]  # (H,)

        # ---------- Gradient for this episode ----------
        if not use_f_reg:
            # Plain REINFORCE: sum_t alive_t * G_t ∇ log π(a_t | s_t)
            def grad_for_step(s, a, G_t, alive_t):
                grad_log = grad_log_pi_fn(theta, s, a)  # (S, A)
                return alive_t * G_t * grad_log

            grads_t = jax.vmap(grad_for_step)(
                states_idx, actions, G, alive
            )  # (H, S, A)
            ep_grad = jnp.sum(grads_t, axis=0)  # (S, A)

        else:
            # f-regularised REINFORCE, matching your numpy fPG.train logic:
            #
            # cumulative_grad_log_pi += grad_log_pi_t
            # grads += γ^t * cumulative_grad_log_pi * r_t
            # grads -= temp * γ^t * cumulative_grad_log_pi * D_f(π(·|s_t)||prior)
            # grads -= temp * γ^t * F(π(·|s_t), s_t)
            def scan_grad(carry, inp):
                cumulative_grad, grads_total, pow_gamma = carry
                s_t, a_t, r_t, alive_t, probs_t = inp

                grad_log_pi_t = grad_log_pi_fn(theta, s_t, a_t)
                grad_log_pi_t = alive_t * grad_log_pi_t  # mask padded steps

                cumulative_grad = cumulative_grad + grad_log_pi_t

                # Divergence and F-vector
                divergence = _f_divergence_tsallis(probs_t, prior, alpha)
                F_t = _f_vector_F(probs_t, prior, alpha, S, A, s_t)
                F_t = alive_t * F_t

                grads_total = (
                    grads_total
                    + pow_gamma
                    * (
                        cumulative_grad * r_t
                        - f_temp * cumulative_grad * divergence
                        - f_temp * F_t
                    )
                )

                pow_gamma = pow_gamma * gamma
                return (cumulative_grad, grads_total, pow_gamma), None

            carry_init = (
                jnp.zeros_like(theta),                # cumulative_grad
                jnp.zeros_like(theta),                # grads_total
                jnp.array(1.0, dtype=jnp.float32),    # pow_gamma = γ^t
            )
            (cumulative_grad, grads_total, _), _ = jax.lax.scan(
                scan_grad,
                carry_init,
                (states_idx, actions, rewards, alive, probs_t),
            )
            del cumulative_grad
            ep_grad = grads_total  # (S, A)

        G0 = G[0]
        return ep_grad, G0

    # ---------- Batch over B episodes ----------
    keys_all = jax.random.split(key, B + 1)
    key_out = keys_all[0]
    episode_keys = keys_all[1:]  # (B,)

    ep_grads, returns0 = jax.vmap(run_one_episode)(episode_keys)  # (B,S,A), (B,)
    grads_J = jnp.mean(ep_grads, axis=0)  # (S, A)
    J_est = jnp.mean(returns0)  # scalar

    # Optional log-barrier regularisation (for softmax / logbarrier)
    if logbarrier_lambda > 0.0:
        def reg_loss(theta_reg):
            def row_loss(theta_s):
                probs_s = policy_row_fn(theta_s)
                probs_s = jnp.clip(probs_s, 1e-12, jnp.inf)
                probs_s = probs_s / jnp.sum(probs_s)
                return -jnp.sum(jnp.log(probs_s))

            return logbarrier_lambda * jax.vmap(row_loss)(theta_reg).sum()

        reg_grad = jax.grad(reg_loss)(theta)
    else:
        reg_grad = jnp.zeros_like(theta)

    # ---------- Update step ----------
    g_E = grads_J - reg_grad  # Euclidean gradient

    if not riemannian:
        # Standard Euclidean gradient ascent
        theta_next = theta + step_size * g_E
    else:
        # Riemannian update on the product of spheres (per-state)
        #
        #   g_R(s) = g_E(s) - (θ_s^T g_E(s)) θ_s
        #   θ_s ← (θ_s + η g_R(s)) / ||θ_s + η g_R(s)||_2
        proj_dot = jnp.sum(theta * g_E, axis=1, keepdims=True)  # (S,1)
        g_R = g_E - proj_dot * theta

        theta_tmp = theta + step_size * g_R
        norms = jnp.linalg.norm(theta_tmp, axis=1, keepdims=True) + 1e-12
        theta_next = theta_tmp / norms

    return key_out, theta_next, J_est


# ---------------------------------------------------------------------
#  High-level factories for each algorithm
# ---------------------------------------------------------------------
def build_pg_algos(
    S: int,
    A: int,
    prior: chex.Array,
) -> Dict[str, Callable]:
    """
    Returns a dict:
      algos[name] -> factory function that takes hyperparams and returns:
        {
          "init_theta": init_theta(key) -> θ (S,A),
          "update_step": update_step(key, θ, env, params, γ, H, B)
        }
    """

    algos: Dict[str, Dict[str, Callable]] = {}

    # ---- 1) fPG: Alpha-divergence softargmax parametrisation + f-regularisation ----
    def make_fpg(alpha: float, step_size: float, f_temp: float):
        # Robustly detect α=1 (KL case)
        use_kl_case = abs(float(alpha) - 1.0) < 1e-8

        if use_kl_case:
            # α=1 → KL case → standard softmax parametrisation
            def policy_row_fn(theta_s: chex.Array) -> chex.Array:
                return softmax_policy_theta_row(theta_s)
        else:
            # Tsallis / Alpha-divergence case (same pattern as PPO)
            param_divergence = AlphaDivergence(
                alpha=float(alpha),
                use_implicit_diff=True,
            )

            def alpha_softargmax(scores: chex.Array) -> chex.Array:
                # scores: (..., A) or (A,)
                flat_scores = scores.reshape(-1, scores.shape[-1])

                def _softarg(v):
                    # v: (A,)
                    return param_divergence.softargmax(v, prior=None, beta=1.0)

                probs = jax.vmap(_softarg)(flat_scores)  # (batch, A)
                return probs.reshape(scores.shape)

            def policy_row_fn(theta_s: chex.Array) -> chex.Array:
                return alpha_softargmax(theta_s)

        # Gradient of log π via autodiff through the policy_row_fn
        _, grad_log_pi_fn = make_logpi_fn(policy_row_fn)

        def init_theta(key):
            return jnp.zeros((S, A))

        update_core = partial(
            pg_update_step,
            policy_row_fn=policy_row_fn,
            grad_log_pi_fn=grad_log_pi_fn,
            step_size=step_size,
            logbarrier_lambda=0.0,
            use_f_reg=True,
            prior=prior,
            alpha=float(alpha),
            f_temp=f_temp,
            riemannian=False,
        )
        # After partial, args: (key, theta, env, params, gamma, H, B)
        # Make env, H, B static
        update = jax.jit(update_core, static_argnums=(2, 5, 6))

        return {"init_theta": init_theta, "update_step": update}

    algos["fpg"] = make_fpg

    # ---- 2) log-barrier softmax PG (Euclidean) ----
    def make_logbarrier(step_size: float, lb_lambda: float):
        def policy_row_fn(theta_s: chex.Array) -> chex.Array:
            return softmax_policy_theta_row(theta_s)

        _, grad_log_pi_fn = make_logpi_fn(policy_row_fn)

        def init_theta(key):
            return jnp.zeros((S, A))

        update_core = partial(
            pg_update_step,
            policy_row_fn=policy_row_fn,
            grad_log_pi_fn=grad_log_pi_fn,
            step_size=step_size,
            logbarrier_lambda=lb_lambda,
            use_f_reg=False,
            riemannian=False,
        )
        update = jax.jit(update_core, static_argnums=(2, 5, 6))

        return {"init_theta": init_theta, "update_step": update}

    algos["logbarrier"] = make_logbarrier

    # ---- 3) Escort-transform PG (Euclidean) ----
    def make_escort(p: float, step_size: float):
        def policy_row_fn(theta_s: chex.Array, p_inner=p) -> chex.Array:
            return escort_policy_theta_row(theta_s, p_inner)

        _, grad_log_pi_fn = make_logpi_fn(policy_row_fn)

        def init_theta(key):
            theta0 = 1e-3 * jax.random.normal(key, (S, A))
            return theta0

        update_core = partial(
            pg_update_step,
            policy_row_fn=policy_row_fn,
            grad_log_pi_fn=grad_log_pi_fn,
            step_size=step_size,
            logbarrier_lambda=0.0,
            use_f_reg=False,
            riemannian=False,
        )
        update = jax.jit(update_core, static_argnums=(2, 5, 6))

        return {"init_theta": init_theta, "update_step": update}

    algos["escort"] = make_escort

    # ---- 4) Hadamard PG with Riemannian update on the sphere ----
    def grad_log_pi_hadamard(theta: chex.Array, state: int, action: int) -> chex.Array:
        """
        Euclidean gradient of log π(a|s) w.r.t θ:

            θ_s ∈ S^{A-1}, π(a|s) = θ_{s,a}^2 (since ||θ_s||=1)
            log π(a|s) = 2 log |θ_{s,a}| ⇒ ∂/∂θ_{s,a} log π(a|s) = 2 / θ_{s,a}.

        Only (state, action) entry is non-zero.
        """
        grad = jnp.zeros_like(theta)
        theta_sa = theta[state, action]
        eps = 1e-8
        grad = grad.at[state, action].set(2.0 / (theta_sa + jnp.sign(theta_sa) * eps))
        return grad

    def make_hadamard(step_size: float):
        def policy_row_fn(theta_s: chex.Array) -> chex.Array:
            return hadamard_policy_theta_row(theta_s)

        def init_theta(key):
            theta_raw = jax.random.normal(key, (S, A))
            norms = jnp.linalg.norm(theta_raw, axis=1, keepdims=True) + 1e-12
            return theta_raw / norms

        update_core = partial(
            pg_update_step,
            policy_row_fn=policy_row_fn,
            grad_log_pi_fn=grad_log_pi_hadamard,
            step_size=step_size,
            logbarrier_lambda=0.0,
            use_f_reg=False,
            riemannian=True,
        )
        update = jax.jit(update_core, static_argnums=(2, 5, 6))

        return {"init_theta": init_theta, "update_step": update}

    algos["hadamard"] = make_hadamard

    return algos
