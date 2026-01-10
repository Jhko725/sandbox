from typing import Any

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
from dynamical_systems.continuous.ode_base import (
    _infer_stepsize_controller,
    AbstractODE,
)
from jaxtyping import Array, Float

from dynamics_discovery.custom_types import FloatScalar
from dynamics_discovery.models.abstract import AbstractDynamicsModel


ODEState = Float[Array, " dim"]
StackedODEState = Float[Array, "time dim"]


class ODEModel(AbstractDynamicsModel, AbstractODE):
    """
    A class that wraps an AbstractODE to provide an AbstractDynamicsModel interface.

    Useful when you want to feed in the ground-truth ODE to evaluation scripts that
    expect objects conforming to the AbstractDynamicsModel interface.
    """

    ode: AbstractODE

    solver: dfx.AbstractSolver = eqx.field(static=True)
    dt0: float | None = eqx.field(static=True)
    rtol: float | None = eqx.field(static=True)
    atol: float | None = eqx.field(static=True)

    def __init__(
        self,
        ode: AbstractODE,
        solver: dfx.AbstractSolver = dfx.Tsit5(),
        dt0: float | None = None,
        rtol: float | None = 1e-4,
        atol: float | None = 1e-6,
    ):
        self.ode = ode
        self.solver = solver
        self.dt0 = dt0
        self.rtol = rtol
        self.atol = atol

    @property
    def dim(self) -> int:
        return self.ode.dim

    def rhs(self, t, u, args):
        return self.ode.rhs(t, u, args)

    def _diffeqsolve(
        self,
        ts: Float[Array, " time"],
        u0: Float[Array, " dim"],
        saveat: dfx.SaveAt,
        **diffeqsolve_kwargs,
    ):
        stepsize_controller = _infer_stepsize_controller(
            self.dt0, self.rtol, self.atol, ts
        )
        return dfx.diffeqsolve(
            dfx.ODETerm(self.rhs),
            self.solver,
            ts[0],
            ts[-1],
            self.dt0,
            u0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            **diffeqsolve_kwargs,
        ).ys

    def step(
        self,
        t0: FloatScalar,
        t1: FloatScalar,
        u0: Float[Array, " dim"],
        args: Any = None,
        **kwargs: Any,
    ) -> Float[Array, " dim"]:
        del args
        ts = jnp.asarray([t0, t1])
        return self._diffeqsolve(ts, u0, saveat=dfx.SaveAt(t1=True), **kwargs)[0]

    def solve(
        self,
        ts: Float[Array, " time"],
        u0: Float[Array, " dim"],
        args: Any = None,
        **kwargs,
    ) -> Float[Array, "time dim"]:
        del args
        return self._diffeqsolve(ts, u0, saveat=dfx.SaveAt(ts=ts), **kwargs)
