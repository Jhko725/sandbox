import equinox as eqx
import jax.numpy as jnp


class Lorenz63(eqx.Module):
    sigma: float = 10
    beta: float = 8 / 3
    rho: float = 28

    @eqx.filter_jit
    def rhs(self, t, u, args):
        del t, args
        x, y, z = u
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        return jnp.stack([dx, dy, dz], axis=0)
