import abc
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


InShape = Float[Array, " *shape_in"]
OutShape = Float[Array, " *shape_out"]


class AbstractInvertible(eqx.Module):
    """
    Abstract base class for models that support an inverse transformation.

    Useful for creating latent dynamics models, where such invertible models
    are used to transform between the input and latent spaces.
    """

    @abc.abstractmethod
    def __call__(self, x_in: InShape) -> OutShape: ...

    @abc.abstractmethod
    def inverse(self, x_out: OutShape) -> InShape: ...


class InvertibleLinear(AbstractInvertible):
    """
    Wraps the `equinox.nn.Linear` layer to provide the inverse transformation.

    The inverse is computed using the Moore-Penrose pseudoinverse and as such,
    works as a right/left inverse, depending on the dimensionality of the input
    and output spaces.
    """

    _linear: eqx.nn.Linear

    def __init__(
        self,
        in_features: int | Literal["scalar"],
        out_features: int | Literal["scalar"],
        use_bias: bool = True,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        self._linear = eqx.nn.Linear(
            in_features, out_features, use_bias, dtype, key=key
        )

    @property
    def weight(self):
        return self._linear.weight

    @property
    def bias(self):
        return self._linear.bias

    def __call__(self, x_in: Float[Array, ""], *, key: PRNGKeyArray | None = None):
        return self._linear(x_in, key=key)

    def inverse(self, x_out, *, key: PRNGKeyArray | None = None):
        """
        Inverse transformation for the linear layer.

        Note the code mirrors the implementation of equinox.nn.Linear.__call__.
        """
        if self._linear.out_features == "scalar":
            if jnp.shape(x_out) != ():
                raise ValueError("x_out must have scalar shape")
            x_out = jnp.broadcast_to(x_out, (1,))

        if self.bias is not None:
            x_out = x_out - self.bias

        x_in = jnp.linalg.pinv(self.weight) @ x_out

        if self._linear.in_features == "scalar":
            assert jnp.shape(x_in) == (1,)
            x_in = jnp.squeeze(x_in)

        return x_in
