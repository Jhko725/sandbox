from itertools import pairwise

import numpy as np
from einops import rearrange
from jaxtyping import ArrayLike, Float
from scipy.interpolate import make_smoothing_spline
from scipy.sparse import csr_array
from scipy.spatial import KDTree
from tqdm import tqdm


def estimate_noise_stdev(
    t: Float[ArrayLike, " t"],
    x: Float[ArrayLike, "t dim"],
    max_allowed_rel_noise: float = 0.4,
) -> Float[np.ndarray, ""]:
    """Use smoothing cubic splines to detrend the data and compute the standard
    deviation of the residual signal (assumed to be noise).

    If the estimated noise stdev >= max_allowed_rel_noise*signal stdev, the estimation
    is considered to have failed and nan is returned instead. This is because the GCV
    criterion used in scipy to auto-determine the smoothing strength can sometimes fail
    and just consider the entire signal as noise (spline becomes a near constant
    function)."""

    spl: Float[ArrayLike, ["t dim"]] = make_smoothing_spline(t, x)
    x_smoothed = spl(t)
    noise_std = np.std(x - x_smoothed, axis=0)
    if np.any(noise_std > max_allowed_rel_noise * np.std(x, axis=0)):
        return np.asarray(np.nan)
    else:
        return np.mean(noise_std)  # ty:ignore[invalid-return-type]


def find_r_min(dataset, max_allowed_rel_noise=0.4, multiplier=3):
    """Determine minimum radius for neighbor selection. This is done by estimating the
    average noise stdev for each trajectory in the dataset, then computing the mean
    (excluding cases where the noise estimation failed). Finally, some constant multiple
    of the average noise strength is used as the minimum radius."""

    noise_stdevs = np.asarray(
        [
            estimate_noise_stdev(t_i, u_i, max_allowed_rel_noise)
            for t_i, u_i in zip(tqdm(dataset.t), dataset.u)
        ]
    )
    return np.nanmean(noise_stdevs) * multiplier


def neighbors_per_point(
    sparse_adjacency_matrix: csr_array,
) -> Float[np.ndarray, " points"]:
    """Given a sparse adjacency matrix of booleans encoding the adjacency information of
     undirected neighbor connectivities (True if (i,j) are neighbors, False if not),
    return the number of neighbors for each point (i.e., the degree of each point)."""
    return sparse_adjacency_matrix.sum(axis=1).A[:, 0]


def find_r_max(
    dataset,
    r_max_init: float,
    num_sweep: int,
    r_min: float = 0.0,
    target_neighborhood_rel_size: float = 0.05,
):
    r"""For each point in the dataset, count number of neighbors between (r_min, r),
    where r \in (r_min, r_max_init). Find and return the radius r* for which the range
    (r_min, r*) on average contains target_neighborhood_rel_size*total number of points.
    """
    tree = KDTree(rearrange(dataset.u, "traj time dim -> (traj time) dim"))

    dist_mat = tree.sparse_distance_matrix(
        tree, max_distance=r_max_init, output_type="coo_matrix"
    ).tocsr()

    sweep_radii = np.linspace(r_min, r_max_init, num_sweep)

    valid_neighbor_counts = neighbors_per_point(dist_mat > sweep_radii[0])
    target_size = target_neighborhood_rel_size * dist_mat.shape[0]
    if np.mean(valid_neighbor_counts) < target_size:
        raise ValueError("""Insufficient number of neighbors in (r_min, r_max_init). 
        Either increase r_max_init, decrease r_min, or decrease 
        target_neighborhood_rel_size and try again.""")

    neighborhood_sizes = [0]
    for r_prev, r in tqdm(pairwise(sweep_radii)):
        mean_neigh_size = np.mean(
            valid_neighbor_counts - neighbors_per_point(dist_mat > r)
        )

        neighborhood_sizes.append(mean_neigh_size)

        if mean_neigh_size >= target_size:
            # Linear interpolation to determine crossing point
            r_target = r_prev + (r - r_prev) * (
                target_size - neighborhood_sizes[-2]
            ) / (mean_neigh_size - neighborhood_sizes[-2])
            break

    else:
        raise ValueError("r_max_init is too small.")

    neighborhood_sizes = np.asarray(neighborhood_sizes) + np.mean(
        neighbors_per_point(dist_mat > 0.0) - valid_neighbor_counts
    )
    return r_target, (
        sweep_radii[: len(neighborhood_sizes)],
        neighborhood_sizes / dist_mat.shape[0],
    )
