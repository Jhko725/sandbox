from collections.abc import Callable

import equinox as eqx
import jax
import diffrax as dfx
from jaxtyping import Float, Array


class NeuralODE(eqx.Module):
    in_size: int
    width_size: int
    depth: int
    out_size: int | None = None
    activation: Callable = jax.nn.gelu
    key: int = 0
    solver: dfx.AbstractSolver = dfx.Tsit5()
    stepsize_controller: dfx.AbstractStepSizeController = dfx.PIDController(
        rtol=1e-4, atol=1e-4
    )
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

    def _set_dt0(self, t) -> Float[Array, ""] | None:
        if isinstance(self.stepsize_controller, dfx.ConstantStepSize):
            dt0 = t[1] - t[0]
        else:
            dt0 = None
        return dt0

    def solve(self, t, u0, args=None, **diffeqsolve_kwargs):
        dt0 = self._set_dt0(t)
        sol = dfx.diffeqsolve(
            dfx.ODETerm(self.rhs),
            self.solver,
            t[0],
            t[-1],
            dt0,
            u0,
            args,
            saveat=dfx.SaveAt(ts=t),
            stepsize_controller=self.stepsize_controller,
            **diffeqsolve_kwargs,
        )
        return sol.ys
