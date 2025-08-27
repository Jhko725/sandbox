from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, PyTree
from lineax.internal import two_norm


def tree_float_precision(tree: PyTree) -> jnp.dtype | None:
    """
    Given a PyTree containing inexact (float/complex) arrays sharing the same precision,
    return the corresponding dtype.

    Currently treats jnp.float32/64 and jnp.complex64/128 as different precisions, which
    will have to be fixed in the future if necessary
    """
    tree_float_dtypes = jax.tree.leaves(
        jax.tree.map(lambda x: x.dtype, eqx.filter(tree, eqx.is_inexact_array))
    )
    dtype_set = set(tree_float_dtypes)

    match list(dtype_set):
        case []:
            return None
        case [dtype]:
            return dtype
        case [_, *_]:
            raise ValueError(
                f"""The given PyTree contains the following 
                multiple floating point dtypes: {dtype_set}"""
            )


def tree_satisfy_float_precision(*trees: PyTree, expect_x64: bool = True) -> bool:
    """
    Test if the given PyTrees all conform to the expected floating point precision.
    """
    tree_dtypes = jax.tree.map(tree_float_precision, trees)
    expected_dtype = jnp.float64 if expect_x64 else jnp.float32
    return jax.tree.all(jax.tree.map(lambda x: x == expected_dtype, tree_dtypes))


def tree_zeros_like(tree: PyTree[ArrayLike, " T"]) -> PyTree[ArrayLike, " T"]:
    return jax.tree.map(jnp.zeros_like, tree)


def tree_scale(scalar: float, tree: PyTree[ArrayLike, " T"]) -> PyTree[ArrayLike, " T"]:
    """Multiply a pytree by a scalar.

    Type promotion is not considered.
    Basically identical to optax's implementation:
    https://github.com/google-deepmind/optax/blob/main/optax/tree_utils/_tree_math.py"""
    return jax.tree.map(lambda x: scalar * x, tree)


def tree_normalize(x: PyTree[ArrayLike, " T"]) -> PyTree[ArrayLike, " T"]:
    """Given a pytree of arrays, compute its normalized version.

    Like the original ConFIG code (https://github.com/tum-pbs/ConFIG/blob/main/conflictfree/grad_operator.py),
    returns a zero vector if the norm of x is zero."""
    x_norm = two_norm(x)
    return jax.lax.cond(x_norm > 0, partial(tree_scale, 1 / x_norm), tree_zeros_like, x)
