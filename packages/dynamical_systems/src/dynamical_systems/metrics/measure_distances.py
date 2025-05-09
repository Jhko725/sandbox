import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from ott.tools import sinkhorn_divergence as ott_sinkhorn


@jax.jit
def maximum_mean_discrepancy(
    p: Float[Array, "batch dim"],
    q: Float[Array, "batch dim"],
    bandwidth: tuple[float, ...] = (0.2, 0.5, 0.9, 1.3),
) -> Array:
    """Maximum Mean Discrepancy.

    Emprical maximum mean discrepancy. The lower the result the more evidence that
    distributions are the same.
    Input arrays are reshaped to dimension: `batch_size x -1`, where `-1`
    indicates that all non-batch dimensions are flattened.
    This implementation was adapted from [1].

    Args:
      p: first sample, distribution P
      q: second sample, distribution Q
      bandwidth: Multiscale levels for the bandwidth.

    Returns:
      mmd value.
    """
    # Samples x and y are of size `batch_size x state_space_dim`, e.g. for Lorenz
    # system `state_space_dim` is `3`, for KS it is `xspan x 1`, for NS it is
    # `h x w x 1`.
    # These arrays are then reshaped to be order two with shape
    # `batch_size x state_space_dim_flattened`.

    xx, yy, zz = jnp.matmul(p, p.T), jnp.matmul(q, q.T), jnp.matmul(p, q.T)
    rx = jnp.broadcast_to(jnp.expand_dims(jnp.diag(xx), axis=0), xx.shape)
    ry = jnp.broadcast_to(jnp.expand_dims(jnp.diag(yy), axis=0), yy.shape)

    dxx = rx.T + rx - 2.0 * xx
    dyy = ry.T + ry - 2.0 * yy
    dxy = rx.T + ry - 2.0 * zz

    xx, yy, xy = (jnp.zeros_like(xx), jnp.zeros_like(xx), jnp.zeros_like(xx))

    # Multiscale bandwisth.
    for a in bandwidth:
        xx += a**2 * (a**2 + dxx) ** -1
        yy += a**2 * (a**2 + dyy) ** -1
        xy += a**2 * (a**2 + dxy) ** -1

    # TODO: We may want to use jnp.sqrt here; see [2].
    return jnp.sqrt(jnp.mean(xx + yy - 2.0 * xy))


@jax.jit
def sinkhorn_divergence(p: Float[Array, "batch dim"], q: Float[Array, "batch dim"]):
    """
    Compute the empirical Sinkhorn divergence between two arrays p, q corresponding to
    samples from two distributions P, Q.

    This quantity is useful because it is well-defined for distributions with
    non-overlapping supports.

    The implementation is from [1], which is part of the original implementation of [2].

    [1] https://github.com/google-research/swirl-dynamics/blob/main/swirl_dynamics/projects/ergodic/measure_distances.py
    [2] Y. Schiff et al. "DySLIM: Dynamics Stable Learning by Invariant Measure for
    Chaotic Systems. International Conference on Machine Learning. PMLR, 2024."
    """
    return ott_sinkhorn.sinkdiv(p, q, static_b=False)[0]
