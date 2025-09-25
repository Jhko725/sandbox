import jax.numpy as jnp
from jaxtyping import Array, Float


def cosine_similarity(
    x_true: Float[Array, "batch time dim"],
    x_pred: Float[Array, "batch time dim"],
    axis: int = 0,
) -> Float[Array, "..."]:
    """
    Compute cosine similarity between two batches of points along an axis.

    The shapes of the two input arrays must be identical at axis dimension
    and broadcastable along all other dimensions.
    """
    if not isinstance(axis, int):
        raise ValueError("axis must only be an integer!")
    x_mean = jnp.mean(x_true, axis=0)
    x = x_true - x_mean
    y = x_pred - x_mean
    xy = jnp.vecdot(x, y, axis=-1)
    x_norm = jnp.linalg.norm(x, axis=-1)
    y_norm = jnp.linalg.norm(y, axis=-1)
    return jnp.mean(xy / (x_norm * y_norm), axis=0)


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


def root_mean_square_error(
    x_true: Float[Array, "batch time dim"],
    x_pred: Float[Array, "batch time dim"],
) -> Float[Array, " time"]:
    """
    Compute cosine similarity between two batches of points along an axis.

    The shapes of the two input arrays must be identical at axis dimension
    and broadcastable along all other dimensions.
    """
    return jnp.mean(jnp.sqrt(jnp.mean((x_true - x_pred) ** 2, axis=-1)), axis=0)


def valid_prediction_time(
    t: Float[Array, " time"],
    x_true: Float[Array, "batch time dim"],
    x_pred: Float[Array, "batch time dim"],
    epsilon: float = 0.2,
):
    rmse: Float[Array, " time"] = root_mean_square_error(x_true, x_pred)
    return t[jnp.argmax(rmse >= epsilon)]
