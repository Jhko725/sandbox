import abc
from typing import Any

import equinox as eqx
import jax
from jaxtyping import Array, Float, PyTree

from ..custom_types import FloatScalar
from ..utils.array import append_to_front


ModelState = PyTree[Float[Array, "*dims"], " T"]
StackedModelState = PyTree[Float[Array, " time *dims"], " T"]
LatentState = PyTree[Float[Array, "*dims_latent"], " T"]
StackedLatentState = PyTree[Float[Array, " time *dims_latent"], " T"]


class AbstractDynamicsModel(eqx.Module, strict=True):
    """
    Abstract base class for all neural network models that aim to model time-series
    data from (partial) observations of a dynamical system.
    """

    @abc.abstractmethod
    def step(
        self,
        t0: FloatScalar,
        t1: FloatScalar,
        u0: ModelState,
        args: Any = None,
        **kwargs: Any,
    ) -> ModelState:
        """
        Given the model state u0 at time t0, generate the model state at some time
        t1>t0.
        """
        ...

    def solve(
        self,
        ts: Float[Array, " time"],
        u0: ModelState,
        args: Any = None,
        **kwargs: Any,
    ) -> StackedModelState:
        """
        Given an array of time points ts, generate the coresponding trajectory of the
        model dynamics us.

        Implicitly, the time points ts are assumed to be monotonically increasing.
        However, this constraint is not enforced, and depending on the model, one could
        make it output predictions for other kinds of ts, such as
        monotonically decreasing.

        The default implementation produces ts by `jax.lax.scan`-ing `self.step` over
        ts.
        Depending on the model, one can override the method in the subclass to provide
        a more efficient implementation.
        This is done, for example, for the neural ode model in `models/neuralode.py`.
        """

        def _step(
            carry: tuple[ModelState, FloatScalar], t1: FloatScalar
        ) -> tuple[tuple[ModelState, FloatScalar], ModelState]:
            """
            Inner function wrapping `self.step` to satisfy the semantics of
            `jax.lax.scan`.
            """
            u0_, t0 = carry
            u1 = self.step(t0, t1, u0_, args, **kwargs)
            carry_new = (u1, t1)
            return carry_new, u1

        carry_init = (u0, ts[0])
        us_: PyTree[Float[Array, " time-1 *dims"], " T"] = jax.lax.scan(
            _step, carry_init, ts[1:]
        )[1]
        return jax.tree.map(append_to_front, u0, us_)


class AbstractLatentDynamicsModel(AbstractDynamicsModel):
    @abc.abstractmethod
    def to_latent(self, u: ModelState) -> LatentState: ...

    @abc.abstractmethod
    def to_obs(self, z: LatentState) -> ModelState: ...

    @abc.abstractmethod
    def latent_step(
        self,
        t0: FloatScalar,
        t1: FloatScalar,
        z0: LatentState,
        args: Any = None,
        **kwargs: Any,
    ) -> LatentState:
        """
        Analogous to `self.step`, but in latent space.
        Given latent state u0 at time t0, generate the latent state at some time
        t1>t0.
        """
        ...

    def step(
        self,
        t0: FloatScalar,
        t1: FloatScalar,
        u0: ModelState,
        args: Any = None,
        **kwargs: Any,
    ) -> ModelState:
        z0 = self.to_latent(u0)
        z1 = self.latent_step(t0, t1, z0, args, **kwargs)
        return self.to_obs(z1)

    def solve(
        self,
        ts: Float[Array, " time"],
        u0: ModelState,
        args: Any = None,
        **kwargs: Any,
    ) -> StackedModelState:
        """
        Given an array of time points ts, generate the coresponding trajectory of the
        model dynamics us.

        Implicitly, the time points ts are assumed to be monotonically increasing.
        However, this constraint is not enforced, and depending on the model, one could
        make it output predictions for other kinds of ts, such as
        monotonically decreasing.

        The default implementation produces ts by `jax.lax.scan`-ing `self.step` over
        ts.
        Depending on the model, one can override the method in the subclass to provide
        a more efficient implementation.
        This is done, for example, for the neural ode model in `models/neuralode.py`.
        """

        def _step(
            carry: tuple[ModelState, FloatScalar], t1: FloatScalar
        ) -> tuple[tuple[ModelState, FloatScalar], ModelState]:
            """
            Inner function wrapping `self.step` to satisfy the semantics of
            `jax.lax.scan`.
            """
            z0_, t0 = carry
            z1 = self.latent_step(t0, t1, z0_, args, **kwargs)
            carry_new = (z1, t1)
            return carry_new, z1

        z0 = self.to_latent(u0)
        carry_init = (z0, ts[0])
        zs_: PyTree[Float[Array, " time-1 *dims"], " T"] = jax.lax.scan(
            _step, carry_init, ts[1:]
        )[1]
        zs = jax.tree.map(append_to_front, z0, zs_)
        return eqx.filter_vmap(self.to_obs)(zs)
