import diffrax as dfx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..continuous import AbstractODE, TangentODE
from ..linalg import gram_schmidt


def lyapunov_gr(
    ode: AbstractODE,
    u0: Float[Array, " dim"],
    t: Float[Array, " time_perturb"],
    solver=dfx.Tsit5(),
    rtol=1e-6,
    atol=1e-6,
    **diffeqsolve_kwargs,
) -> Float[Array, "dim time_perturb"]:
    ode_tangent = TangentODE(ode)
    u0_tangent = (u0, jnp.identity(ode.dim))
    log_norm_sum = jnp.zeros(ode.dim)

    carry = t[0], u0_tangent, log_norm_sum

    def _inner(carry, t1):
        t0, u0_tangent_, log_norm_sum0 = carry
        u1, u1_tangent = dfx.diffeqsolve(
            dfx.ODETerm(ode_tangent.rhs),
            solver,
            t0,
            t1,
            None,
            u0_tangent_,
            None,
            saveat=dfx.SaveAt(t1=True),
            stepsize_controller=dfx.PIDController(rtol=rtol, atol=atol),
            **diffeqsolve_kwargs,
        ).ys
        u1_tangent_gr, u1_tangent_norm = gram_schmidt(u1_tangent[0])
        log_norm_sum1 = log_norm_sum0 + jnp.log(u1_tangent_norm)
        carry_new = t1, (u1[0], u1_tangent_gr), log_norm_sum1
        return carry_new, (log_norm_sum1, u1[0])

    _, (log_norm_sums, u_vals) = jax.lax.scan(_inner, carry, t[1:])
    log_norm_sums = jnp.concatenate(
        [jnp.expand_dims(log_norm_sum, 0), log_norm_sums], axis=0
    )
    u_vals = jnp.concatenate([jnp.expand_dims(u0, 0), u_vals], axis=0)
    return log_norm_sums.T / t, u_vals
