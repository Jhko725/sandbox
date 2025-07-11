import abc

import diffrax as dfx
import equinox as eqx
from jaxtyping import Array, Float


class AbstractODE(eqx.Module):
    """Abstract base class for dynamical systems governed by ordinary differential
    equations.

    Ordinary differential equations represented by subclasses of AbstractODE are meant
    to be numerically solved using either `diffrax.diffeqsolve` or the `solve_ode`
    function defined in this module.
    """

    dim: eqx.AbstractVar[int]

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


def _infer_stepsize_controller(dt0, rtol, atol):
    match dt0, rtol, atol:
        case _, float(), float():
            return dfx.PIDController(rtol, atol)
        case float(), None, None:
            return dfx.ConstantStepSize()
        case _:
            raise ValueError("""Unexpected combination of arguments: either 
                             (rtol, atol) = (float, float) or 
                             dt0 = float and (rtol, atol) = (None, None)""")


@eqx.filter_jit
def solve_ode(
    ode: AbstractODE,
    ts: Float[Array, " time"],
    u0: Float[Array, " *batch dim"],
    solver: dfx.AbstractSolver = dfx.Tsit5(),
    dt0: float | None = None,
    rtol: float | None = 1e-4,
    atol: float | None = 1e-4,
    **diffeqsolve_kwargs,
) -> Float[Array, "*batch time dim"]:
    """A convenience wrapper function around diffrax.diffeqsolve."""

    # TODO: maybe some sort of shape check between expected shape of ode.rhs and u0?
    stepsize_controller = _infer_stepsize_controller(dt0, rtol, atol)

    sol = dfx.diffeqsolve(
        dfx.ODETerm(ode.rhs),
        solver,
        ts[0],
        ts[-1],
        dt0,
        u0,
        saveat=dfx.SaveAt(ts=ts),
        stepsize_controller=stepsize_controller,
        **diffeqsolve_kwargs,
    )
    return sol.ys
