from itertools import product

import numpy as np
import scipy.spatial as scspatial
from einops import rearrange
from jaxtyping import Float
from tqdm import tqdm

from .data import TimeSeriesDataset


def get_tangent_space_projector(
    du: Float[np.ndarray, "neighbors dim"],
    proj_dim: int = 2,
) -> Float[np.ndarray, "dim proj_dim"]:
    vh = np.linalg.svd(du, full_matrices=False)[2]
    return vh.T[:, :proj_dim]


def least_squares(
    X: Float[np.ndarray, "M N"], Y: Float[np.ndarray, "M K"]
) -> Float[np.ndarray, "N K"]:
    """For the linear regression problem XB=Y, calculate the least squares solution
    via the pseudoinverse"""
    return np.linalg.pinv(X) @ Y


def total_least_squares(
    X: Float[np.ndarray, "M N"], Y: Float[np.ndarray, "M K"]
) -> Float[np.ndarray, "N K"]:
    """For the linear regression problem XB=Y, calculate the total least squares
    solution by computing the SVD of the augmented matrix [X Y].

    See https://en.wikipedia.org/wiki/Total_least_squares#Algebraic_point_of_view for
    details."""
    XY: Float[np.ndarray, "M N+K"] = np.concatenate((X, Y), axis=1)
    _, _, Vh = np.linalg.svd(XY, full_matrices=True)
    V = np.conjugate(Vh).T
    k = X.shape[1]
    Vxx, Vyy = V[:k, k:], V[k:, k:]
    return -Vxx @ np.linalg.pinv(Vyy)


def estimate_pushforward_matrices(
    dataset: TimeSeriesDataset,
    radius: float,
    dim_project: int,
    num_neighbor_threshold: int,
):
    maps1, maps2, weights = [], [], []
    tree = scspatial.KDTree(
        rearrange(dataset.u[:, :-1], "trajs time dim -> (trajs time) dim")
    )
    for traj_ind, time_ind in tqdm(
        product(range(dataset.u.shape[0]), range(dataset.u.shape[1] - 1))
    ):
        u0, u1 = dataset.u[traj_ind, time_ind], dataset.u[traj_ind, time_ind + 1]
        neigh_inds0 = np.divmod(
            np.asarray(tree.query_ball_point(u0, radius)), dataset.u.shape[1] - 1
        )
        neigh_inds1 = np.divmod(
            np.asarray(tree.query_ball_point(u1, radius)), dataset.u.shape[1] - 1
        )
        if (
            len(neigh_inds0[0]) < num_neighbor_threshold
            or len(neigh_inds1[0]) < num_neighbor_threshold
        ):
            maps1.append(np.zeros_like(u0, shape=(u0.shape[-1], dim_project)))
            maps2.append(np.zeros_like(u0, shape=(dim_project, u0.shape[-1])))
        else:
            du0: Float[np.ndarray, "neighbors dim"] = dataset.u[*neigh_inds0] - u0
            du0_next = dataset.u[neigh_inds0[0], neigh_inds0[1] + 1] - u1
            proj0: Float[np.ndarray, "dim dim_proj"] = get_tangent_space_projector(
                du0, dim_project
            )
            proj1: Float[np.ndarray, "dim dim_proj"] = get_tangent_space_projector(
                dataset.u[*neigh_inds1] - u1, dim_project
            )

            du0_proj = du0 @ proj0
            du0_next_proj = du0_next @ proj1
            A: Float[np.ndarray, "dim_proj dim_proj"] = least_squares(
                du0_proj, du0_next_proj
            )
            maps1.append(proj0)
            maps2.append(A @ proj1.T)
        weights.append(len(neigh_inds0[0]))
    maps1: Float[np.ndarray, "trajs*time-1 dim dim_proj"] = np.stack(maps1, axis=0)
    maps2: Float[np.ndarray, "trajs*time-1 dim_proj dim"] = np.stack(maps2, axis=0)
    maps1 = rearrange(
        maps1,
        "(trajs time) dim dim_proj -> trajs time dim dim_proj",
        trajs=dataset.u.shape[0],
    )
    maps2 = rearrange(
        maps2,
        "(trajs time) dim_proj dim -> trajs time dim_proj dim",
        trajs=dataset.u.shape[0],
    )
    weights = rearrange(
        np.asarray(weights),
        "(trajs time) -> trajs time",
        trajs=dataset.u.shape[0],
    )
    return maps1, maps2, weights
