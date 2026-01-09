from collections.abc import Callable
from typing import Any

import diffrax as dfx
import equinox as eqx
import jax
from dynamical_systems.continuous.ode_base import (
    _infer_stepsize_controller,
    AbstractODE,
)
from jaxtyping import Array, Float

from ..custom_types import FloatScalar, PRNGKeyArrayLike
from .abstract import AbstractDynamicsModel


ODEState = Float[Array, " dim"]
StackedODEState = Float[Array, "time dim"]


class NeuralODE(AbstractDynamicsModel, AbstractODE):
    net: eqx.nn.MLP
    dim: int = eqx.field(static=True)
    width: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)
    solver: dfx.AbstractSolver = eqx.field(static=True)
    stepsize_controller: dfx.AbstractStepSizeController = eqx.field(static=True)
    dt0: float | None = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        width: int,
        depth: int,
        activation: Callable = jax.nn.gelu,
        solver: dfx.AbstractSolver = dfx.Tsit5(),
        dt0: float | None = None,
        rtol: float | None = 1e-4,
        atol: float | None = 1e-4,
        *,
        key: PRNGKeyArrayLike = 0,
    ):
        self.dim = dim
        self.width = width
        self.depth = depth
        self.activation = activation
        self.net = eqx.nn.MLP(
            in_size=self.dim,
            out_size=self.dim,
            width_size=self.width,
            depth=self.depth,
            activation=self.activation,
            key=jax.random.PRNGKey(key),
        )
        self.solver = solver
        self.stepsize_controller = _infer_stepsize_controller(dt0, rtol, atol)
        self.dt0 = dt0

    def rhs(self, t, u, args):
        del t, args
        return self.net(u)

    def _diffeqsolve(self, t0, t1, u0, saveat: dfx.SaveAt, **diffeqsolve_kwargs):
        return dfx.diffeqsolve(
            dfx.ODETerm(self.rhs),
            self.solver,
            t0,
            t1,
            self.dt0,
            u0,
            saveat=saveat,
            stepsize_controller=self.stepsize_controller,
            **diffeqsolve_kwargs,
        ).ys

    def step(
        self,
        t0: FloatScalar,
        t1: FloatScalar,
        u0: ODEState,
        args: Any = None,
        **kwargs: Any,
    ) -> ODEState:
        del args
        return self._diffeqsolve(t0, t1, u0, saveat=dfx.SaveAt(t1=True), **kwargs)[0]

    def solve(
        self, ts: Float[Array, " time"], u0: ODEState, args: Any = None, **kwargs
    ) -> StackedODEState:
        del args
        return self._diffeqsolve(
            ts[0], ts[-1], u0, saveat=dfx.SaveAt(ts=ts), max_steps=16384, **kwargs
        )
