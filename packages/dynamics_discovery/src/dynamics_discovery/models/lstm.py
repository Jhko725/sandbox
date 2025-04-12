import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array

LSTMState = tuple[Float[Array, " dim"], Float[Array, " dim"]]


class LSTM(eqx.Module):
    dim: int
    key: int = 0
    dim_control: int = 1
    use_bias: bool = True
    rnncell: eqx.nn.LSTMCell = eqx.field(init=False)

    def __post_init__(self):
        self.rnncell = eqx.nn.LSTMCell(
            self.dim_control,
            self.dim,
            use_bias=self.use_bias,
            key=jax.random.PRNGKey(self.key),
        )

    def rhs(self, t, state: LSTMState, args) -> LSTMState:
        del t, args
        return self.rnncell(jnp.zeros(1), state)

    def make_initial_state(self, u0: Float[Array, " self.dim"]) -> LSTMState:
        return (u0, jnp.zeros_like(u0))

    def solve(self, t, u0, args=None):
        pass
