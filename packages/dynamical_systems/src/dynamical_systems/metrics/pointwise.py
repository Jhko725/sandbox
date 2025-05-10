import jax.numpy as jnp
from jaxtyping import Array, Float


def cosine_similarity(
    x: Float[Array, "..."],
    y: Float[Array, "..."],
    axis: int = 0,
) -> Float[Array, "..."]:
    """
    Compute cosine similarity between two batches of points along an axis.

    The shapes of the two input arrays must be identical at axis dimension
    and broadcastable along all other dimensions.
    """
    if not isinstance(axis, int):
        raise ValueError("axis must only be an integer!")

    xy = jnp.vecdot(x, y, axis=axis)
    x_norm = jnp.linalg.norm(x, axis=axis)
    y_norm = jnp.linalg.norm(y, axis=axis)
    return xy / (x_norm * y_norm)


def mean_squared_error(
    x: Float[Array, "..."],
    y: Float[Array, "..."],
    axis: int | tuple[int, ...] = 0,
) -> Float[Array, "..."]:
    """
    Compute the mean squared error between two batches of points along an axis.

    The shapes of the two input arrays must be identical at axis dimension
    and broadcastable along all other dimensions.

    `axis` can be an integer or tuple of integers if reduction along multiple dimensions
    is desired.
    """
    return jnp.mean((x - y) ** 2, axis=axis)
