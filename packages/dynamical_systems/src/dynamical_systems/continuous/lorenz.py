from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from .ode_base import AbstractODE


class Lorenz63(AbstractODE):
    sigma: float = 10
    beta: float = 8 / 3
    rho: float = 28
    dim: ClassVar[int] = 3

    @eqx.filter_jit
    def rhs(
        self, t: Float[Array, ""], u: Float[Array, " {self.dim}"], args=None
    ) -> Float[Array, " {self.dim}"]:
        del t, args
        x, y, z = u
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        return jnp.stack([dx, dy, dz], axis=0)

    def jacobian(self, t: Float[Array, ""], u: Float[Array, " self.dim"], args=None):
        del t, args
        x, y, z = u
        return jnp.asarray(
            [[-self.sigma, self.sigma, 0], [self.rho - z, -1, -x], [y, x, -self.beta]]
        )


class Lorenz96(AbstractODE):
    dim: int = 6
    F: float = 10.0

    @eqx.filter_jit
    def rhs(self, t, u, args=None):
        del t, args
        return (jnp.roll(u, 1) - jnp.roll(u, -2)) * jnp.roll(u, -1) - u + self.F
