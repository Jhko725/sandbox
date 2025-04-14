import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from ..nn.invertible import AbstractInvertible, InvertibleLinear
from ..nn.rnn import AbstractRNNCell, RNNCell


RNNState = Float[Array, " dim_latent"]


class LatentRNNBase(eqx.Module):
    rnncell: eqx.AbstractVar[AbstractRNNCell]
    latent2obs: eqx.AbstractVar[AbstractInvertible]

    def to_latent(
        self, u: Float[Array, " self.dim"]
    ) -> Float[Array, " self.dim_latent"]:
        return self.latent2obs.inverse(u)

    def to_obs(self, h: Float[Array, " self.dim_latent"]) -> Float[Array, " self.dim"]:
        return self.latent2obs(h)

    def rhs(self, t, state: RNNState, args=None) -> RNNState:
        del t
        control = jnp.zeros(1) if self.dim_control is None else args[0]
        return self.rnncell(control, state)

    def make_initial_state(self, u0: Float[Array, " self.dim"]) -> RNNState:
        return self.to_latent(u0)

    def solve(
        self, ts: Float[Array, " time"], u0: Float[Array, " self.dim"], args=None
    ):
        # Currently, doesn't support external imputs
        # Assume t is equi-spaced
        state0 = self.make_initial_state(u0)

        def body_fn(state_prev, t):
            state_next = self.rhs(t, state_prev, args)
            return state_next, state_next

        hs_: Float[Array, " time-1 self.dim_latent"] = jax.lax.scan(
            body_fn, state0, ts[1:]
        )[1]
        h0 = jnp.expand_dims(state0, axis=0)
        hs = jnp.concatenate((h0, hs_), axis=0)
        return eqx.filter_vmap(self.to_obs)(hs)


class LatentRNN(LatentRNNBase):
    rnncell: eqx.Module
    latent2obs: AbstractInvertible
    dim: int = eqx.field(static=True)
    dim_latent: int = eqx.field(static=True)
    dim_control: int | None = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        dim_latent: int = 20,
        dim_control: int | None = None,
        *,
        key: PRNGKeyArray | int = 0,
    ):
        self.dim = dim
        self.dim_latent = dim_latent
        self.dim_control = dim_control

        key_rnn, key_latent = jax.random.split(jax.random.PRNGKey(key), 2)
        input_size = 1 if self.dim_control is None else self.dim_control
        self.rnncell = RNNCell(input_size, dim_latent, use_bias=True, key=key_rnn)
        self.latent2obs = InvertibleLinear(
            dim_latent, dim, use_bias=True, key=key_latent
        )


LSTMState = tuple[Float[Array, " dim"], Float[Array, " dim"]]


class LatentLSTM(LatentRNNBase):
    """LSTM model from Mikhaeil et al. NeurIPS (2022), where a LSTM is used to model the
    dynamics in latent space, and the input and latent space are connected via an
    affine transform."""

    dim: int = eqx.field(static=True)
    dim_latent: int = eqx.field(static=True, init=20)
    key: int = 0
    dim_control: int = 1
    use_bias: bool = True
    latent2obs: InvertibleLinear = eqx.field(init=False)
    rnncell: eqx.nn.LSTMCell = eqx.field(init=False)

    def __init__(
        self,
        dim: int,
        dim_latent: int = 20,
        dim_control: int | None = None,
        *,
        key: PRNGKeyArray | int = 0,
    ):
        self.dim = dim
        self.dim_latent = dim_latent
        self.dim_control = dim_control

        key_rnn, key_latent = jax.random.split(jax.random.PRNGKey(key), 2)
        input_size = 1 if self.dim_control is None else self.dim_control
        self.rnncell = eqx.nn.LSTMCell(
            input_size, dim_latent, use_bias=True, key=key_rnn
        )
        self.latent2obs = InvertibleLinear(
            dim_latent, dim, use_bias=True, key=key_latent
        )

    def make_initial_state(self, u0: Float[Array, " self.dim"]) -> LSTMState:
        h0 = self.to_latent(u0)
        return (h0, jnp.zeros_like(h0))

    def solve(
        self, ts: Float[Array, " time"], u0: Float[Array, " self.dim"], args=None
    ):
        # Assume t is equi-spaced
        state0 = self.make_initial_state(u0)

        def body_fn(state_prev, t):
            state_next = self.rhs(t, state_prev, args)
            return state_next, state_next[0]

        hs_ = jax.lax.scan(body_fn, state0, ts[1:])[
            1
        ]  #: Float[Array, " time-1 self.dim_latent"]
        h0 = jnp.expand_dims(state0[0], axis=0)
        hs = jnp.concatenate((h0, hs_), axis=0)
        return eqx.filter_vmap(self.to_obs)(hs)
