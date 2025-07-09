import abc

import equinox as eqx
from jaxtyping import Array, Float


class AbstractBijection(eqx.Module):
    """Abstract class to represent a bijection

    Currently, very naive, supporting only the forward and inverse interfaces."""

    @abc.abstractmethod
    def __call__(self, x: Array) -> Array: ...

    @abc.abstractmethod
    def inverse(self, y: Array) -> Array: ...


class ShiftScaleTransform(AbstractBijection):
    shift: Float[Array, " dim"]
    scale: Float[Array, " dim"]

    def __call__(self, x: Float[Array, "*batch dim"]) -> Float[Array, "*batch dim"]:
        return (x - self.shift) / self.scale

    def inverse(self, y: Float[Array, "*batch dim"]) -> Float[Array, "*batch dim"]:
        return y * self.scale + self.shift
