import jax.numpy as jnp
from jaxtyping import Array, Float


def append_to_front(
    x0: Float[Array, "*dims"], x_rest: Float[Array, " num-1 *dims"]
) -> Float[Array, " num *dims"]:
    """
    Append an array to a stacked sequence of arrays.
    """
    x0_ = jnp.expand_dims(x0, axis=0)
    return jnp.concatenate((x0_, x_rest), axis=0)
