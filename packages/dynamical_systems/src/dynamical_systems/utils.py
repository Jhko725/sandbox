import inspect
from typing import Any

from jaxtyping import ArrayLike


def get_name(obj):
    if inspect.isfunction(obj) or inspect.isclass(obj):
        return obj.__qualname__
    else:
        return repr(obj)


def is_arraylike_scalar(x: Any) -> bool:
    return isinstance(x, ArrayLike) and x.shape == ()
