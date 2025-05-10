from functools import partial

import jax
import jax.numpy as jnp
import scipy.spatial as scspatial
from jaxtyping import Array, Float


def empirical_one_step_jacobian(
    us: Float[Array, "time dim"], num_neighbors: int = 25
) -> Float[Array, "time-1 dim dim"]:
    """
    Method to estimate the jacobian of an observed time series, from the works of
    Abarbanel and others.
    Currently, only the first order method is supported.
    """
    u0, u1 = us[:-1], us[1:]
    kdtree = scspatial.KDTree(u0)
    # Index [:, 1:] used to exclude the query point itself
    idx_nn = kdtree.query(u0, k=num_neighbors + 1)[1][:, 1:]
    idx_nn = jnp.asarray(idx_nn, dtype=jnp.int_)

    @partial(jax.vmap, in_axes=(0, 0, 0))
    def jac_one_step(u0_, u1_, idx_nn_):
        du0 = jnp.take(us, idx_nn_, axis=0) - u0_
        du1 = jnp.take(us, idx_nn_ + 1, axis=0) - u1_
        return (jnp.linalg.pinv(du0) @ du1).T

    return jac_one_step(u0, u1, idx_nn)
