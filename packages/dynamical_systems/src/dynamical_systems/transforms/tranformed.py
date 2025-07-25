import equinox as eqx

from ..continuous import AbstractODE
from .bijections import AbstractBijection


class TransformedODE(AbstractODE):
    """For a base ode dv/dt = f(v) and a bijection u = g(v),
    represent the ODE for u: du/dt = grad(g)*f(v)"""

    ode: AbstractODE
    bijection: AbstractBijection

    def rhs(self, t, u, args=None):
        v = self.bijection.inverse(u)
        _, dv = eqx.filter_jvp(self.bijection, (v,), (self.ode.rhs(t, v, args),))
        return dv

    @property
    def dim(self):
        return self.ode.dim
