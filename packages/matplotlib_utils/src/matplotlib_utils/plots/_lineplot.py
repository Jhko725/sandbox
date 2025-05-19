from jaxtyping import ArrayLike, Float

from ..custom_types import Axes


def plot_line_and_band(
    ax: Axes,
    x: Float[ArrayLike, " N"],
    y: Float[ArrayLike, " N"],
    y_widths: Float[ArrayLike, " N"]
    | tuple[Float[ArrayLike, " N"], Float[ArrayLike, " N"]],
    color: str = "royalblue",
    alpha: float = 1.0,
    alpha_band: float = 0.3,
    **plot_kwargs,
) -> Axes:
    ax.plot(x, y, color=color, alpha=alpha, **plot_kwargs)
    if isinstance(y_widths, tuple):
        width_lower, width_upper = y_widths
    else:
        width_lower, width_upper = y_widths, y_widths

    ax.fill_between(x, y - width_lower, y + width_upper, color=color, alpha=alpha_band)
    return ax
