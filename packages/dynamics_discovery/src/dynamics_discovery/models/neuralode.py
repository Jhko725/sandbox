from collections.abc import Callable

import equinox as eqx
import jax


class NeuralODE(eqx.Module):
    in_size: int
    width_size: int
    depth: int
    out_size: int | None = None
    activation: Callable = jax.nn.gelu
    key: int = 0
    net: eqx.nn.MLP = eqx.field(init=False)
    dim: int = eqx.field(init=False)

    def __post_init__(self):
        if self.out_size is None:
            self.out_size = self.in_size
        self.net = eqx.nn.MLP(
            in_size=self.in_size,
            out_size=self.out_size,
            width_size=self.width_size,
            depth=self.depth,
            activation=self.activation,
            key=jax.random.PRNGKey(self.key),
        )
        self.dim = self.out_size

    def rhs(self, t, u, args):
        del t, args
        return self.net(u)
