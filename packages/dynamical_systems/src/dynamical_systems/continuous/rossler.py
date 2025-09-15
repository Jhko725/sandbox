from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from .ode_base import AbstractODE


class Rossler76(AbstractODE):
    a: float = 0.2
    b: float = 0.2
    c: float = 5.7
    dim: ClassVar[int] = 3

    @eqx.filter_jit
    def rhs(
        self, t: Float[Array, ""], u: Float[Array, " {self.dim}"], args=None
    ) -> Float[Array, " {self.dim}"]:
        del t, args
        x, y, z = u
        dx = -y - z
        dy = x + self.a * y
        dz = self.b + z * (x - self.c)
        return jnp.stack([dx, dy, dz], axis=0)

    def jacobian(
        self, t: Float[Array, ""], u: Float[Array, " {self.dim}"], args=None
    ) -> Float[Array, "{self.dim} {self.dim}"]:
        del t, args
        x, _, z = u
        return jnp.asarray(
            [
                [0, -1, -1],
                [1, self.a, 0],
                [z, 0, x - self.c],
            ]
        )


class Rossler79(AbstractODE):
    a: float = 0.25
    b: float = 3.0
    c: float = 0.5
    d: float = 0.05
    dim: ClassVar[int] = 4

    @eqx.filter_jit
    def rhs(
        self, t: Float[Array, ""], u: Float[Array, " {self.dim}"], args=None
    ) -> Float[Array, " {self.dim}"]:
        del t, args
        x, y, z, w = u
        dx = -y - z
        dy = x + self.a * y + w
        dz = self.b + x * z
        dw = -self.c * z + self.d * w
        return jnp.stack([dx, dy, dz, dw], axis=0)

    def jacobian(
        self, t: Float[Array, ""], u: Float[Array, " {self.dim}"], args=None
    ) -> Float[Array, "{self.dim} {self.dim}"]:
        del t, args
        x, _, z, _ = u
        return jnp.asarray(
            [[0, -1, -1, 0], [1, self.a, 0, 1], [z, 0, x, 0], [0, 0, -self.c, self.d]]
        )
