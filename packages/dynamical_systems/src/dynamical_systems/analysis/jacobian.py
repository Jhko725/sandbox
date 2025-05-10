from functools import partial

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import scipy.spatial as scspatial
from jaxtyping import Array, Float


@eqx.filter_jit
def jacobian(
    ode,
    t: Float[Array, " batch"],
    u: Float[Array, "batch dim"],
) -> Float[Array, "batch dim dim"]:
    if hasattr(ode, "jacobian"):
        jacobian_fn = ode.jacobian
    else:

        def _rhs(u_: Float[Array, " dim"], t_: Float[Array, ""]):
            return ode.rhs(t_, u_, None)

        def jacobian_fn(t_, u_):
            return eqx.filter_jacrev(_rhs)(u_, t_)

    return eqx.filter_vmap(jacobian_fn, in_axes=(0, 0))(t, u)


def one_step_jacobian(
    ode,
    t: Float[Array, " batch"],
    u: Float[Array, "batch dim"],
    dt: float,
    solver: dfx.AbstractAdaptiveSolver = dfx.Tsit5(),
    rtol: float = 1e-7,
    atol: float = 1e-7,
    **diffeqsolve_kwargs,
) -> Float[Array, "batch dim dim"]:
    @eqx.filter_jacrev
    def _step_jacobian(
        u0: Float[Array, " dim"], t0: Float[Array, ""]
    ) -> Float[Array, " dim"]:
        sol = dfx.diffeqsolve(
            dfx.ODETerm(ode.rhs),
            solver,
            t0,
            t0 + dt,
            None,
            u0,
            stepsize_controller=dfx.PIDController(rtol=rtol, atol=atol),
            args=None,
            **diffeqsolve_kwargs,
        )
        return sol.ys[0]

    return eqx.filter_vmap(_step_jacobian, in_axes=(0, 0))(u, t)


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
