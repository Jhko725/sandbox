from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from ..custom_types import FloatScalar, PRNGKeyArrayLike
from ..nn.invertible import AbstractInvertible, InvertibleLinear
from ..nn.rnn import AbstractRNNCell, RNNCell
from .abstract import AbstractLatentDynamicsModel


ModelState = Float[Array, " dim"]
LatentState = Float[Array, " dim_latent"]


class LatentRNN(AbstractLatentDynamicsModel):
    rnncell: AbstractRNNCell
    latent2obs: AbstractInvertible
    dim: int = eqx.field(static=True)
    dim_latent: int = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        dim_latent: int = 60,
        *,
        key: PRNGKeyArrayLike = 0,
    ):
        self.dim = dim
        self.dim_latent = dim_latent

        key_rnn, key_latent = jax.random.split(jax.random.PRNGKey(key), 2)
        self.rnncell = RNNCell(1, dim_latent, use_bias=True, key=key_rnn)
        self.latent2obs = InvertibleLinear(
            dim_latent, dim, use_bias=True, key=key_latent
        )

    def to_latent(self, u: ModelState) -> LatentState:
        return self.latent2obs.inverse(u)

    def to_obs(self, z: LatentState) -> ModelState:
        return self.latent2obs(z)

    def latent_step(
        self,
        t0: FloatScalar,
        t1: FloatScalar,
        z0: LatentState,
        args: Any = None,
        **kwargs: Any,
    ) -> LatentState:
        # Assumes autonomous dynamics for now
        del t0, t1, args, kwargs
        return self.rnncell(jnp.zeros(1), z0)


LSTMLatentState = tuple[Float[Array, " dim_latent"], Float[Array, " dim_latent"]]


class LatentLSTM(AbstractLatentDynamicsModel):
    """LSTM model from Mikhaeil et al. NeurIPS (2022), where a LSTM is used to model the
    dynamics in latent space, and the input and latent space are connected via an
    affine transform."""

    latent2obs: InvertibleLinear
    rnncell: eqx.nn.LSTMCell
    dim: int = eqx.field(static=True)
    dim_latent: int = eqx.field(static=True, init=20)

    def __init__(
        self,
        dim: int,
        dim_latent: int = 30,
        *,
        key: PRNGKeyArray | int = 0,
    ):
        self.dim = dim
        self.dim_latent = dim_latent

        key_rnn, key_latent = jax.random.split(jax.random.PRNGKey(key), 2)
        self.rnncell = eqx.nn.LSTMCell(1, dim_latent, use_bias=True, key=key_rnn)
        self.latent2obs = InvertibleLinear(
            dim_latent, dim, use_bias=True, key=key_latent
        )

    def to_latent(self, u: ModelState) -> LSTMLatentState:
        hidden_state = self.latent2obs.inverse(u)
        cell_state = jnp.zeros_like(hidden_state)
        return (hidden_state, cell_state)

    def to_obs(self, z: LSTMLatentState) -> ModelState:
        hidden_state, _ = z
        return self.latent2obs(hidden_state)

    def latent_step(
        self,
        t0: FloatScalar,
        t1: FloatScalar,
        z0: LSTMLatentState,
        args: Any = None,
        **kwargs: Any,
    ) -> LSTMLatentState:
        # Assumes autonomous dynamics for now
        del t0, t1, args, kwargs
        return self.rnncell(jnp.zeros(1), z0)
