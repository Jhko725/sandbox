import abc

import equinox as eqx
from jaxtyping import ArrayLike, Float


class AbstractBijection(eqx.Module):
    """Abstract class to represent a bijection

    Currently, very naive, supporting only the forward and inverse interfaces."""

    @abc.abstractmethod
    def __call__(self, x: ArrayLike) -> ArrayLike: ...

    @abc.abstractmethod
    def inverse(self, y: ArrayLike) -> ArrayLike: ...


class Standardize(AbstractBijection):
    mean: Float[ArrayLike, " dim"]
    std: Float[ArrayLike, " dim"]

    def __call__(
        self, x: Float[ArrayLike, "*batch dim"]
    ) -> Float[ArrayLike, "*batch dim"]:
        return (x - self.mean) / self.std

    def inverse(
        self, y: Float[ArrayLike, "*batch dim"]
    ) -> Float[ArrayLike, "*batch dim"]:
        return y * self.std + self.mean
