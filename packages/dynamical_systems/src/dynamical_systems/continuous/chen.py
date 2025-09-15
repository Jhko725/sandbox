from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from .ode_base import AbstractODE


class Chen99(AbstractODE):
    a: float = 35
    b: float = 3
    c: float = 28
    dim: ClassVar[int] = 3

    @eqx.filter_jit
    def rhs(
        self, t: Float[Array, ""], u: Float[Array, " {self.dim}"], args=None
    ) -> Float[Array, " {self.dim}"]:
        del t, args
        x, y, z = u
        dx = self.a * (y - x)
        dy = (self.c - self.a) * x - x * z + self.c * y
        dz = x * y - self.b * z
        return jnp.stack([dx, dy, dz], axis=0)

    def jacobian(self, t: Float[Array, ""], u: Float[Array, " self.dim"], args=None):
        del t, args
        x, y, z = u
        return jnp.asarray(
            [
                [-self.a, self.a, 0],
                [self.c - self.a - z, self.c, -x],
                [y, x, -self.b],
            ]
        )


class HyperChen05(AbstractODE):
    a: float = 35
    b: float = 3
    c: float = 12
    d: float = 7
    r: float = 0.58
    dim: ClassVar[int] = 4

    @eqx.filter_jit
    def rhs(
        self, t: Float[Array, ""], u: Float[Array, " {self.dim}"], args=None
    ) -> Float[Array, " {self.dim}"]:
        del t, args
        x, y, z, w = u
        dx = self.a * (y - x) + w
        dy = (self.d - z) * x + self.c * y
        dz = x * y - self.b * z
        dw = y * z + self.r * w
        return jnp.stack([dx, dy, dz, dw], axis=0)

    def jacobian(self, t: Float[Array, ""], u: Float[Array, " self.dim"], args=None):
        del t, args
        x, y, z, _ = u
        return jnp.asarray(
            [
                [-self.a, self.a, 0, 1],
                [self.d - z, self.c, -x, 0],
                [y, x, -self.b, 0],
                [0, z, y, self.r],
            ]
        )
