r"""F-divergences and associated operators.

The classical softplus, logistic sigmoid, softmax and softargmax operators
can be seen as maximizing a variational objective using a KL divergence
regularization. This module implements new generalizations of these operators
by replacing the KL divergence with an f-divergence, a general class of
divergences that includes the KL divergence as a special case.

For more details about these divergences, see:
  * https://en.wikipedia.org/wiki/KL_divergence
  * https://en.wikipedia.org/wiki/F-divergence

Our notation is summarized as follows:
  * f: generating function of the f-divergence.
  * f': derivative of f.
  * f*: convex conjugate of f, where f*(v) := \sup_u u * v - f(u).
  * f*': derivative of the convex conjugate of f.
  * u: variable in dom(f).
  * v: variable in dom(f*).
  * p: a positive measure.
  * q: a positive measure.
  * q(0): importance of the negative class.
  * q(1): importance of the positive class.
  * s: input to f-softplus or f-sigmoid.
  * beta: regularization strength.
  * tau: root of the f-sigmoid function.

The f-divergence between two discrete positive measures p and q is defined as:
  f_div(p, q) := \sum_i f(p(i) / q(i)) * q(i).

We define the f-softplus operator as:
  f_softplus(s; q) := max_{pi in [0,1]} pi * s - f_div((1 - pi, pi), q).

We define the f-sigmoid operator as:
  f_sigmoid(s; q) := argmax_{pi in [0,1]} pi * s - f_div((1 - pi, pi), q).

A mathematical derivation shows that:
  f_softplus(s; q) = tau + q(0) * f*(-tau) + q(1) * f*(s - tau)
  f_softargmax(s; q) = q(1) * f*'(s - tau),

where tau is the root of:
  q(0) f*'(-tau) + q(1) f*'(s - tau) = 1.

We can further introduce a regularization strength parameter beta:
  f_softplus(s; beta, q) :=
    max_{pi in [0,1]} pi * s - beta * f_div((1 - pi, pi), q)
  f_sigmoid(s; beta, q) :=
    argmax_{pi in [0,1]} pi * s - beta * f_div((1 - pi, pi), q).

It can easily be shown that:
  f_softplus(s; beta, q) = beta * f_softplus(s / beta; q)
  f_sigmoid(s; beta, q) = f_sigmoid(s / beta; q).

We can define a binary classification loss as
  f_binary_loss(scores, labels; beta, q)
    := f_softplus(scores; beta, q) +
       beta * f_div((1 - pi, pi), q) -
       scores * labels.

"""

import abc
from collections.abc import Callable
from typing import Any
import functools

import chex
import jax
import jax.numpy as jnp


_DEFAULT_PRIOR_NEG: float = 0.5  # Default importance of the negative class.
_DEFAULT_PRIOR_POS: float = 0.5  # Default importance of the positive class.


class FDivergence(metaclass=abc.ABCMeta):
  """Abstract class for f-divergences and associated operators."""

  def __init__(self, num_iter=30, verbose=False, use_implicit_diff=False):
    self.num_iter = num_iter
    self.verbose = verbose
    self.use_implicit_diff = use_implicit_diff

  @property
  def sparse(self) -> bool:
    """Whether the sigmoid and softargmax can be sparse.

    If True, we apply a thresholding in the root finding problem. This
    thresholding is unnecessary and ill-defined when sparse=False.
    """
    return False

  @abc.abstractmethod
  def generating_function(self, u: chex.Array) -> jax.Array:
    """Generating function of the f-divergence."""

  @abc.abstractmethod
  def generating_function_derivative(self, u: chex.Array) -> jax.Array:
    """Derivative of the generative function `f`."""

  @abc.abstractmethod
  def generating_function_conjugate(self, v: chex.Array) -> jax.Array:
    """Convex conjugate of the function `f`."""

  @abc.abstractmethod
  def generating_function_conjugate_derivative(
      self, v: chex.Array
  ) -> jax.Array:
    """Derivative of the convex conjugate of the function `f`."""

  def divergence(self, p: chex.Array, q: chex.Array) -> chex.Numeric:
    r"""f-divergence between two positive measures.

    .. math::

      f_div(p, q) := \sum_i f(p(i) / q(i)) * q(i).

    Args:
      p: a positive measure, shape [batch_size, num_classes] or [num_classes].
      q: a positive measure, same shape as `p`.

    Returns:
      An array of [batch_size] or a scalar.

    References:
      * https://en.wikipedia.org/wiki/F_divergence
    """
    return jnp.sum(self.generating_function(p / q) * q, axis=-1)

  def _root_sigmoid(self, s, q0=_DEFAULT_PRIOR_NEG, q1=_DEFAULT_PRIOR_POS):
    s = jnp.vstack((jnp.zeros_like(s), s)).T
    q = jnp.array([q0, q1])
    return self._root_softargmax(s, q)

  def _thresholded_value(self, diff, q):
    if self.sparse:
      zeros = jnp.zeros_like(diff)
      diff = jnp.maximum(diff, self.generating_function_derivative(zeros))
    return q * self.generating_function_conjugate(diff)

  def _thresholded_proba(self, diff, q):
    if self.sparse:
      zeros = jnp.zeros_like(diff)
      diff = jnp.maximum(diff, self.generating_function_derivative(zeros))
    return q * self.generating_function_conjugate_derivative(diff)


  def _root_softargmax(self, s: jax.Array, q: jax.Array) -> chex.Numeric:
    """Root of the softargmax function."""

    def value_fn(tau, s, q):
      if len(s.shape) > 1:
        tau = tau[..., jnp.newaxis]
      q = jnp.broadcast_to(q, s.shape)
      got = jnp.sum(
          self._thresholded_proba(s - tau, q),
          axis=-1,
      )
      error = 1.0 - got
      return error

    def bracket_fn(s, q):
      s_argmax = jnp.argmax(s, keepdims=True, axis=-1)
      s_max = jnp.take_along_axis(s, s_argmax, axis=-1)
      q_s_argmax = jnp.take_along_axis(
          jnp.broadcast_to(q, s.shape), s_argmax, axis=-1
      )
      tau_min = s_max - self.generating_function_derivative(1 / q_s_argmax)
      tau_min = jax.lax.stop_gradient(tau_min)
      tau_max = s_max - self.generating_function_derivative(1 / jnp.sum(q))
      tau_max = jax.lax.stop_gradient(tau_max)
      # Squeeze the last dimension from tau_max and tau_min.
      # For single inputs, this ensures the output is a scalar.
      # For batched inputs, this converts the shape from (batch_size, 1)
      # to (batch_size,).
      return jnp.squeeze(tau_min, axis=-1), jnp.squeeze(tau_max, axis=-1)

    bisection_fn = make_bisection(
        value_fn,
        bracket_fn,
        sign=1,  # The function is increasing.
        num_iter=self.num_iter,
        verbose=self.verbose,
        use_implicit_diff=self.use_implicit_diff,
    )
    return bisection_fn(s, q)

  def _softmax(self, s: jax.Array, q: jax.Array) -> jax.Array:
    tau_star = jax.lax.stop_gradient(self._root_softargmax(s, q))
    if len(s.shape) > 1:
      tau_star_ = tau_star[..., jnp.newaxis]
    else:
      tau_star_ = tau_star

    q = jnp.broadcast_to(q, s.shape)
    return tau_star + jnp.sum(
        self._thresholded_value(s - tau_star_, q),
        axis=-1,
    )

  def softmax(
      self,
      scores: jax.Array,
      prior: jax.Array | None = None,
      beta: float = 1.0,
  ) -> jax.Array:
    """(beta*f)-softmax operator.

    Args:
      scores: scores produced by the model, shape [batch_size, num_classes] or
        [num_classes].
      prior: prior probabilities of the classes, shape [num_classes].
      beta: a temperature parameter.

    Returns:
      A scalar.
    """
    if prior is None:
      k = scores.shape[-1]
      prior = jnp.ones(k) / k
    max_scores = jax.lax.stop_gradient(jnp.max(scores, axis=-1))
    scores = scores - jnp.expand_dims(max_scores, axis=-1)
    scores = scores / beta
    return beta * self._softmax(scores, prior) + max_scores

  def _softargmax(self, s: jax.Array, q: jax.Array) -> jax.Array:
    tau_star = self._root_softargmax(s, q)
    if len(s.shape) > 1:
      tau_star = tau_star[..., jnp.newaxis]
    q = jnp.broadcast_to(q, s.shape)
    return self._thresholded_proba(s - tau_star, q)

  def softargmax(
      self,
      scores: jax.Array,
      prior: jax.Array | None = None,
      beta: float = 1.0,
  ) -> jax.Array:
    """(beta*f)-softargmax operator.

    Args:
      scores: scores produced by the model, shape [batch_size, num_classes] or
        [num_classes].
      prior: prior probabilities of the classes, shape [num_classes].
      beta: a temperature parameter.

    Returns:
      An array of the same shape as `scores`.
    """
    if prior is None:
      k = scores.shape[-1]
      prior = jnp.ones(k) / k
    max_scores = jax.lax.stop_gradient(jnp.max(scores, axis=-1, keepdims=True))
    scores = scores - max_scores
    scores = scores / beta
    return self._softargmax(scores, prior)


##############################################################################
# Alpha div example


def alpha_log(u: chex.Array, alpha: float = 1.0) -> jax.Array:
  """Alpha logarithm."""
  # if alpha == 1.0:
  #   return jnp.log(u)
  # else:
  alpha_m1 = alpha - 1.0
  return (jnp.power(u, alpha_m1) - 1) / alpha_m1


def alpha_exp(v: chex.Array, alpha: float = 1.0) -> jax.Array:
  """Alpha exponential."""
  # if alpha == 1.0:
  #   return jnp.exp(v)
  # else:
  alpha_m1 = alpha - 1.0
  return jnp.power(1 + alpha_m1 * v, 1./ alpha_m1)


class AlphaDivergence(FDivergence):
  """Alpha divergence and associated operators."""

  def __init__(
      self, num_iter=30, verbose=False, use_implicit_diff=False, alpha=1.0
  ):
    super().__init__(
        num_iter=num_iter, verbose=verbose, use_implicit_diff=use_implicit_diff
    )
    self.alpha = alpha

  @property
  def sparse(self) -> bool:
    return False

  def generating_function(self, u: chex.Array) -> jax.Array:
    """Generating function (`f`) of the divergence."""
    # if self.alpha == 1.0:
    #   return jax.scipy.special.xlogy(u, u) - (u - 1)
    # else:
    return (jnp.power(u, self.alpha) - 1 - self.alpha * (u - 1)) / (
        self.alpha * (self.alpha - 1)
    )

  def generating_function_derivative(self, u: chex.Array) -> jax.Array:
    """Derivative of the function `f`."""
    return alpha_log(u, alpha=self.alpha)

  def generating_function_conjugate(self, v: chex.Array) -> jax.Array:
    """Convex conjugate of the function `f`."""
    # if self.alpha == 1.0:
    #   return jnp.exp(v) - 1.0
    # else:
    u = alpha_exp(v, alpha=self.alpha)
    # TODO(mblondel): check if we can simplify this expression.
    return u * v - self.generating_function(u)

  def generating_function_conjugate_derivative(
      self, v: chex.Array
  ) -> jax.Array:
    """Derivative of the convex conjugate of the function `f`."""
    return alpha_exp(v, alpha=self.alpha)


##############################################################################
# Bisection


FLAG_SIGN = (
    "Sign is 0, not guaranteed to converge."
)


def _get_sign(value_fn, lower, upper, *args, **kwargs):
  lower_value = value_fn(lower, *args, **kwargs)
  upper_value = value_fn(upper, *args, **kwargs)

  # sign = 1: f(lower) < 0 and f(upper) >= 0.
  # sign = -1: f(lower) > 0 and f(upper) <= 0.
  # sign = 0: the algorithm is not guaranteed to converge.
  return jnp.where((lower_value < 0) & (upper_value >= 0),
                   1,
                   jnp.where((lower_value > 0) & (upper_value <= 0), -1, 0))


def make_bisection(
    value_fn: Callable[..., chex.Numeric],
    bracket_fn: Callable[..., tuple[chex.Numeric, chex.Numeric]],
    sign: int | None = None,
    num_iter: int = 30,
    verbose: bool = False,
    use_implicit_diff: bool = True,
    debug: bool = False,
    use_scan: bool = True,
) -> Callable[..., chex.Numeric]:
  """Make a bisection function.

  Args:
    value_fn: objective. Function of the form value_fn(x, *args, **kwargs).
    bracket_fn: returns the lower and upper bounds of the search space.
      Function of the form bracket_fn(*args, **kwargs).
    sign: whether the function globally increases (1) or globally decreases
      (-1) on the interval returned by bracket_fn, with at least one root in
      between. If None, the sign is detected automatically at the cost of two
      extra calls to value_fn.
    num_iter: number of iterations.
    verbose: whether to print the error at each iteration.
    use_implicit_diff: whether to add implicit differentiation support.
    debug: whether to raise an error if the root is potentially not contained in
      the bracketed function.

  Returns:
    function of the form bisection(*args) if use_implicit_diff is True and
    bisection(*args, **kwargs) if use_implicit_diff is
    False.
  """

  def run(*args, **kwargs):
    lower, upper = bracket_fn(*args, **kwargs)

    sign_ = (
        _get_sign(value_fn, lower, upper, *args, **kwargs)
        if sign is None
        else sign
    )
    if debug:
      def print_error():
        jax.debug.print(
            FLAG_SIGN
        )
      jax.lax.cond(sign_ == 0, print_error, lambda: None)

    mid = 0.5 * (lower + upper)

    # We use a native Python loop without stopping criterion on purpose
    # to avoid branching, so as to make code run faster on TPU.
    if use_scan:
      def scan_body(carry, it):
        lower, upper, mid = carry
        value = value_fn(mid, *args, **kwargs)
        too_large = sign_ * value > 0
        upper = jnp.where(too_large, mid, upper)
        lower = jnp.where(too_large, lower, mid)
        mid = jnp.where(value == 0, mid, 0.5 * (lower + upper))

        if verbose:
          error = jnp.abs(value)
          jax.debug.print("Bisection {it}: {error}", it=it, error=error)
        return (lower, upper, mid), None
      (lower, upper, mid), _ = jax.lax.scan(
          scan_body, (lower, upper, mid), jnp.arange(1, num_iter + 1)
      )
    else:
      for it in range(1, num_iter + 1):
        value = value_fn(mid, *args, **kwargs)
        too_large = sign_ * value > 0
        upper = jnp.where(too_large, mid, upper)
        lower = jnp.where(too_large, lower, mid)
        mid = jnp.where(value == 0, mid, 0.5 * (lower + upper))

        if verbose:
          error = jnp.abs(value)
          jax.debug.print("Bisection {it}: {error}", it=it, error=error)

    return mid

  if use_implicit_diff:
    run = add_implicit_diff_1d(run, value_fn)

  return run


############################################################################
# Implicit_diff


def _absolute_clip(x, eps):
  """Clip x to be at least eps in absolute value."""
  return jnp.where(jnp.absolute(x) < eps, eps * jnp.sign(x + eps), x)


def _jvp_root(value_fn, root, args, tangent):
  """JVP in the first argument of value_fn."""
  # We close over the arguments.
  fn = lambda x: value_fn(x, *args)
  return jax.jvp(fn, (root,), (tangent,))[1]


def _jvp_args(value_fn, root, args, tangents):
  """JVP in the second argument of value_fn."""
  # We close over the solution.
  fn = lambda *y: value_fn(root, *y)
  return jax.jvp(fn, args, tangents)[1]


def root_1d_jvp(
    value_fn: Callable[..., chex.Numeric],
    root: chex.Numeric,
    args: tuple[Any, ...],
    tangents: tuple[Any, ...],
) -> chex.Numeric:
  """Compute the JVP of a root.

  Args:
    value_fn: objective function of the form value_fn(x, *args).
    root: solution of the root finding problem: value_fn(root, *args) == 0.
      Can be a scalar or a 1d array (batch setting).
    args: arguments of value_fn w.r.t. which we want to compute the JVP.
    tangents: a tuple of the same size as len(args). Each tangents[i]
      has the same structure as args[i].

  Returns:
    jvp: same structure as root
  """
  if len(args) != len(tangents):
    raise ValueError("args and tangents should be tuples of the same length.")

  if jnp.isscalar(root):
    deriv_root = jax.grad(value_fn)(root, *args)
  else:  # Batch setting.
    deriv_root = _jvp_root(value_fn, root, args, jnp.ones_like(root))

  jvp_args = _jvp_args(value_fn, root, args, tangents)

  # See equation (2) in https://arxiv.org/abs/2105.15183
  # Both jvp_args and deriv_root have the same structure as `root``.
  eps = jnp.finfo(deriv_root).eps

  stabilized_deriv_root = _absolute_clip(deriv_root, eps)

  return -jvp_args / stabilized_deriv_root


# We can add support for **kwargs by using the same approach as in JAXOPT.
# https://github.com/google/jaxopt/blob/main/jaxopt/_src/implicit_diff.py#L173
def add_implicit_diff_1d(
    root_fn: Callable[..., chex.Numeric], value_fn: Callable[..., chex.Numeric]
) -> Callable[..., chex.Numeric]:
  """Add implicit differentiation to a root finding function.

  Args:
    root_fn: the root finding function to wrap. Function of the form
      root_fn(*args).
    value_fn: the objective function of the root finding problem. Function of
      the form value_fn(x, *args).

  Returns:
    root finding function with implicit differentiation support.
  """
  root_fn_wrapped = jax.custom_jvp(root_fn)

  def root_jvp(primals, tangents):
    root = root_fn(*primals)
    return root, root_1d_jvp(value_fn, root, primals, tangents)

  root_fn_wrapped.defjvp(root_jvp)

  return root_fn_wrapped