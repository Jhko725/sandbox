import diffrax as dfx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..continuous import AbstractODE, TangentODE
from ..linalg import gram_schmidt


def lyapunov_naive(
    ode: AbstractODE,
    u0: Float[Array, " dim"],
    t: Float[Array, " time_perturb"],
    solver=dfx.Tsit5(),
    rtol=1e-6,
    atol=1e-6,
    **diffeqsolve_kwargs,
) -> tuple[Float[Array, "time_perturb dim"], Float[Array, "time_perturb dim"]]:
    """Evaluate the local Lyapunov exponent by naively integrating the tangent dynamics.

    For long time calculations, this method is unstable and result in unreliable
    estimate of the Lyapunov exponents.
    """
    ode_tangent = TangentODE(ode)
    u0_tangent = (u0, jnp.identity(ode.dim))

    u, u_tangent = dfx.diffeqsolve(
        dfx.ODETerm(ode_tangent.rhs),
        solver,
        t[0],
        t[-1],
        None,
        u0_tangent,
        None,
        saveat=dfx.SaveAt(ts=t),
        stepsize_controller=dfx.PIDController(rtol=rtol, atol=atol),
        **diffeqsolve_kwargs,
    ).ys

    eigvals = jnp.linalg.svdvals(u_tangent)

    Dt = jnp.expand_dims(t - t[0], axis=-1)
    lya = jnp.log(eigvals) / Dt

    return lya, u


def lyapunov_gr(
    ode: AbstractODE,
    u0: Float[Array, " dim"],
    t: Float[Array, " time_perturb"],
    solver=dfx.Tsit5(),
    rtol=1e-6,
    atol=1e-6,
    **diffeqsolve_kwargs,
) -> tuple[Float[Array, "time_perturb dim"], Float[Array, "time_perturb dim"]]:
    """Evaluate the local Lyapunov exponent by integrating the tangent dynamics
    and performing Gram-Schmidt orthonormalization every time step to prevent the
    the tangent vectors from collapsing onto the largest eigenvector direction."""
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
        jax.debug.print("Time t0={t0}", t0=t0)
        return carry_new, (log_norm_sum1, u1[0])

    _, (log_norm_sums, u_vals) = jax.lax.scan(_inner, carry, t[1:])
    log_norm_sums = jnp.concatenate(
        [jnp.expand_dims(log_norm_sum, 0), log_norm_sums], axis=0
    )
    u_vals = jnp.concatenate([jnp.expand_dims(u0, 0), u_vals], axis=0)
    Dt = jnp.expand_dims(t - t[0], axis=-1)
    return log_norm_sums / Dt, u_vals


def perturb(
    u0: Float[Array, " dim"],
    norm_perturb: float,
    num_perturbs: int,
    key: jax.random.PRNGKey,
) -> Float[Array, "num_perturb dim"]:
    p = jax.random.normal(key, (num_perturbs, len(u0)))
    p_normed = p / jnp.linalg.norm(p, axis=-1, keepdims=True)
    return u0 + norm_perturb * p_normed


def lyapunov_nonlinear(
    ode: AbstractODE,
    u0: Float[Array, " dim"],
    t: Float[Array, " time_perturb"],
    solver=dfx.Tsit5(),
    rtol=1e-6,
    atol=1e-6,
    norm_perturb: float = 1e-6,
    num_perturbs: int = 25,
    key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    **diffeqsolve_kwargs,
) -> tuple[Float[Array, " time_perturb"], Float[Array, "time_perturb dim"]]:
    """Evaluate the local Lyapunov exponent by integrating trajectories from
    perturbed initial positions, then calculating the divergence of trajectories
    over time.

    Since the original nonlinear dynamics instead of the linearized tangent dynamics
    is integrated, this quantity is referred to as the nonlinear local Lyapunov
    exponents (NLLEs).

    Note that due to the nature of the computation, this method is much more memory and
    compute intensive than the other methods.
    """

    @jax.vmap
    def diffeqsolve_batch(u0):
        u = dfx.diffeqsolve(
            dfx.ODETerm(ode.rhs),
            solver,
            t[0],
            t[-1],
            None,
            u0,
            None,
            saveat=dfx.SaveAt(ts=t),
            stepsize_controller=dfx.PIDController(rtol=rtol, atol=atol),
            **diffeqsolve_kwargs,
        ).ys
        return u

    u0_total: Float[Array, "n_perturbs+1 dim"] = jnp.concatenate(
        (jnp.expand_dims(u0, 0), perturb(u0, norm_perturb, num_perturbs, key)), axis=0
    )

    u_total: Float[Array, "n_perturbs+1 time_perturb dim"] = diffeqsolve_batch(u0_total)
    u, u_perturbs = u_total[0], u_total[1:]

    error_norm = jnp.mean(jnp.linalg.norm(u_perturbs - u, axis=-1), axis=0)
    lya = jnp.log(error_norm / norm_perturb) / (t - t[0])
    return lya, u
