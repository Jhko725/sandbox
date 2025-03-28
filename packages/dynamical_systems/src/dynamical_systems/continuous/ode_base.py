import abc

import diffrax as dfx
import equinox as eqx
from jaxtyping import Array, Float


class AbstractODE(eqx.Module):
    """Abstract base class for dynamical systems governed by ordinary differential
    equations.
    """

    dim: eqx.AbstractClassVar[int]

    @abc.abstractmethod
    def rhs(self, t: Float[Array, ""], u: Float[Array, " {self.dim}"], args=None):
        """Describes the right hand side of the differential equation.

        The method signature is chosen to match the requirements for diffrax.
        """
        ...

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        args = ", ".join([f"{k}={v}" for k, v in vars(self).items()])
        return f"{cls}({args})"


@eqx.filter_jit
def solve_ode(
    ode: AbstractODE,
    t: Float[Array, " time"],
    u0: Float[Array, " *batch dim"],
    solver: dfx.AbstractAdaptiveSolver = dfx.Tsit5(),
    rtol: float = 1e-4,
    atol: float = 1e-4,
    **diffeqsolve_kwargs,
) -> Float[Array, "*batch time dim"]:
    """A convenience wrapper function around diffrax.diffeqsolve."""

    assert u0.shape[-1] == ode.dim, f"""Shape of the initial condition u0: (..., dim) = 
        {u0.shape} must match the dimensionality of the ode = {ode.dim}"""

    sol = dfx.diffeqsolve(
        dfx.ODETerm(ode.rhs),
        solver,
        t[0],
        t[-1],
        None,
        u0,
        saveat=dfx.SaveAt(ts=t),
        stepsize_controller=dfx.PIDController(rtol=rtol, atol=atol),
        **diffeqsolve_kwargs,
    )
    return sol.ys
