import jax.numpy as jnp
from jaxtyping import Array, Float


def relative_error_norm(
    y_pred: Float[Array, " batch *rest"],
    y_true: Float[Array, " batch *rest"],
    axis: int | tuple[int, int] = -1,
    ord: int | None = None,
) -> Float[Array, ""]:
    """
    Calculate the relative error between predicted and true values for scalar, vector,
    and matrix valued data using vector/matrix norm.
    """
    norm_true = jnp.linalg.norm(y_true, ord=ord, axis=axis)
    norm_err = jnp.linalg.norm(y_pred - y_true, ord=ord, axis=axis)
    return norm_err / norm_true
