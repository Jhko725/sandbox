from collections.abc import Sequence

from .custom_types import Axes, Axes3D


def set_labels(ax: Axes, labels: Sequence[str], **kwargs) -> Axes:
    if isinstance(ax, Axes3D):
        n_axis = 3
        label_fns = [ax.set_xlabel, ax.set_ylabel, ax.set_zlabel]
    else:
        n_axis = 2
        label_fns = [ax.set_xlabel, ax.set_ylabel]

    if n_labels := len(labels) > n_axis:
        raise ValueError(
            f"Number of labels ({n_labels})) exceed the number of axes ({n_axis})"
        )

    for set_label_fn, label in zip(label_fns, labels):
        set_label_fn(label, **kwargs)
    return ax
