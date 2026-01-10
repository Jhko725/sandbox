from collections.abc import Callable
from typing import Any

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from dynamical_systems.continuous.ode_base import (
    _infer_stepsize_controller,
    AbstractODE,
)
from jaxtyping import Array, Float

from ..custom_types import FloatScalar, PRNGKeyArrayLike
from .abstract import AbstractDynamicsModel


class NeuralODE(AbstractDynamicsModel, AbstractODE):
    net: eqx.nn.MLP

    dim: int = eqx.field(static=True)
    solver: dfx.AbstractSolver = eqx.field(static=True)
    dt0: float | None = eqx.field(static=True)
    rtol: float | None = eqx.field(static=True)
    atol: float | None = eqx.field(static=True)

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
        self.net = eqx.nn.MLP(
            in_size=dim,
            out_size=dim,
            width_size=width,
            depth=depth,
            activation=activation,
            key=jax.random.PRNGKey(key),
        )
        self.dim = dim
        self.solver = solver
        self.dt0 = dt0
        self.atol = atol
        self.rtol = rtol

    def rhs(
        self, t: FloatScalar, u: Float[Array, " dim"], args=None
    ) -> Float[Array, " dim"]:
        del t, args
        return self.net(u)

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
