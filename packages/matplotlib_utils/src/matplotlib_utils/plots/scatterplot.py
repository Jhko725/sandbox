from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import ArrayLike, Float


def colored_scatterplot(
    data: Sequence[Float[ArrayLike, "dim points"]],
    colors: Sequence[Float[ArrayLike, " points"]],
    figsize: tuple[int, int] = (10, 5),
    colorbar_pad: float = 0.15,
    colorbar_width: float = 0.02,
    **scatter_kwargs,
):
    n_plots = len(data)
    if n_plots != len(colors):
        raise ValueError("Number of colors must match the number of data")

    data_dim = {d.shape[0] for d in data}
    if len(data_dim) > 1:
        raise ValueError("Data arrays must all have same dimensionality")

    match data_dim.pop():
        case 2:
            subplot_kw = dict()
        case 3:
            subplot_kw = {"projection": "3d"}
        case _:
            raise ValueError("Dimension of the data arrays must either be 2 or 3")

    fig, axes = plt.subplots(1, n_plots, figsize=figsize, subplot_kw=subplot_kw)

    if "vmax" not in scatter_kwargs:
        scatter_kwargs["vmax"] = max([np.max(c) for c in colors])

    if "vmin" not in scatter_kwargs:
        scatter_kwargs["vmin"] = min([np.min(c) for c in colors])

    for ax, data_i, c_i in zip(axes, data, colors):
        plot = ax.scatter(*data_i, c=c_i, **scatter_kwargs)

    cax = fig.add_axes(
        [
            axes[-1].get_position().x1 + colorbar_pad,
            axes[-1].get_position().y0,
            colorbar_width,
            axes[-1].get_position().height,
        ]
    )
    fig.colorbar(plot, ax=axes[-1], cax=cax)
    return fig
