import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree


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
