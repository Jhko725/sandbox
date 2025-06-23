import abc

import diffrax as dfx
import equinox as eqx
from jaxtyping import Array, Float


class AbstractODE(eqx.Module):
    """Abstract base class for dynamical systems governed by ordinary differential
    equations.
    """

    dim: eqx.AbstractVar[int]
    default_solver: eqx.AbstractClassVar[dfx.AbstractAdaptiveSolver]
    default_rtol: eqx.AbstractClassVar[float]
    default_atol: eqx.AbstractClassVar[float]

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

    @property
    def stepsize_controller(self):
        return dfx.PIDController(rtol=self.default_rtol, atol=self.default_atol)

    @property
    def solver(self):
        return self.default_solver

    @property
    def dt0(self):
        return None

    def solve(
        self,
        t: Float[Array, " time"],
        u0: Float[Array, " {self.dim}"],
        solver=None,
        rtol=None,
        atol=None,
        **diffeqsolve_kwargs,
    ):
        solver = self.default_solver if solver is None else solver
        rtol = self.default_rtol if rtol is None else rtol
        atol = self.default_atol if atol is None else atol

        sol = dfx.diffeqsolve(
            dfx.ODETerm(self.rhs),
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

    # assert u0.shape[-1] == ode.dim, f"""Shape of the initial condition
    # u0: (..., dim) = {u0.shape} must match the dimensionality of the
    # ode = {ode.dim}"""

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
