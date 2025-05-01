from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import matplotlib.pyplot as plt

from ..custom_types import Axes


Param = TypeVar("Param")
Common = TypeVar("Common")


def comparisonplot(
    param_values: Sequence[Param],
    params_common: Common,
    *,
    plot_fn: Callable[[Axes, Param, Common], None],
    title_fn: Callable[[Param, Common], str],
    **subplots_kw: Any,
):
    fig, axes = plt.subplots(1, len(param_values), **subplots_kw)
    for ax, p in zip(axes, param_values):
        plot_fn(ax, p, params_common)
        if title_fn is not None:
            ax.set_title(title_fn(p, params_common))
    return fig
