from functools import partial

import jax
from jaxtyping import Array, Float

from .ode_base import AbstractODE


class TangentODE(AbstractODE):
    ode: AbstractODE

    @property
    def dim(self) -> int:
        return self.ode.dim * (self.ode.dim + 1)

    def rhs(self, t, u: tuple[Float[Array, " dim"], Float[Array, "dim dim"]], args):
        x, Tx = u

        def rhs_ode(x_):
            return self.ode.rhs(t, x_, args)

        @partial(jax.vmap, in_axes=(None, -1), out_axes=(None, -1))
        def rhs_jac(x_, Tx_i):
            return jax.jvp(rhs_ode, (x_,), (Tx_i,))

        dx, dTx = rhs_jac(x, Tx)

        return dx, dTx

    @property
    def default_solver(self):
        return self.ode.default_solver

    @property
    def default_atol(self):
        return self.ode.default_atol

    @property
    def default_rtol(self):
        return self.ode.default_rtol
