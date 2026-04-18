from collections.abc import Sequence

from ..custom_types import Axes


def plot_violin(
    ax: Axes,
    data,
    positions: Sequence[float],
    *,
    facecolor,
    edgecolor,
    linewidth: float = 1.0,
    alpha: float = 1.0,
    **violinplot_kwargs,
):
    parts = ax.violinplot(data, positions, **violinplot_kwargs)
    for patch in parts["bodies"]:
        patch.set(
            facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha
        )
    if "cbars" in parts:
        parts["cbars"].set(color="black", linewidth=0.5)

    for k in ("cmeans", "cmins", "cmaxes", "cmedians", "cquantiles"):
        if k in parts:
            parts[k].set(color=edgecolor)
    return ax
