from typing import ClassVar

import jax.numpy as jnp

from .ode_base import AbstractODE


class FitzhughNagumo(AbstractODE):
    dim: ClassVar[int] = 2
    a: float = 0.7
    b: float = 0.8
    c: float = 3
    z: float = -0.4

    def rhs(self, t, u, args=None):
        del t, args
        x, y = u
        dx = self.c * (y + x - x**3 / 3 + self.z)
        dy = -(x - self.a + self.b * y) / self.c
        return jnp.stack([dx, dy], axis=0)
