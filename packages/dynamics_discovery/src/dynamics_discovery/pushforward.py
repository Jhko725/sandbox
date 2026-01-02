from itertools import product

import numpy as np
import scipy.spatial as scspatial
from einops import rearrange
from jaxtyping import Float, Int
from tqdm import tqdm

from .data import TimeSeriesDataset


def get_tangent_space_projector(
    du: Float[np.ndarray, "neighbors dim"],
    proj_dim: int = 2,
) -> Float[np.ndarray, "dim proj_dim"]:
    _, s, vh = np.linalg.svd(du, full_matrices=False)
    s2 = s * s
    score = np.sum(s2[proj_dim:]) / np.sum(s2)
    return vh.T[:, :proj_dim], score


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
    rollout: int = 1,
):
    # Construct KDTree for efficient neighbor search
    u_flat = rearrange(dataset.u, "trajs time dim -> (trajs time) dim")
    dim = u_flat.shape[1]
    tree = scspatial.KDTree(u_flat)

    # Estimate tangent space for each point in dataset
    tangent_projectors, scores = [], []
    for u_i in tqdm(u_flat):
        ind_neighbors = tree.query_ball_point(u_i, radius)

        # Handle insufficient neighbors case - fill with dummy values
        if len(ind_neighbors) < num_neighbor_threshold:
            tangent_projectors.append(np.zeros_like(u_i, shape=(dim, dim_project)))
            scores.append(-np.ones_like(u_i, shape=()))  # Fill with -1s
        else:
            du_i = u_flat[ind_neighbors] - u_i
            projector_i, score_i = get_tangent_space_projector(du_i, dim_project)
            tangent_projectors.append(projector_i)
            scores.append(score_i)

    tangent_projectors = rearrange(
        np.asarray(tangent_projectors),
        "(trajs time) dim dim_proj -> trajs time dim dim_proj",
        trajs=dataset.u.shape[0],
    )
    scores = rearrange(
        np.asarray(scores),
        "(trajs time)-> trajs time",
        trajs=dataset.u.shape[0],
    )

    # Compute mapping between tangent spaces
    maps = []
    for traj_ind, time_ind in tqdm(
        product(range(dataset.u.shape[0]), range(dataset.u.shape[1] - rollout))
    ):
        us: Float[np.ndarray, "rollout+1 dim"] = dataset.u[
            traj_ind, time_ind : time_ind + rollout + 1
        ]
        ind_neighbors = tree.query_ball_point(us[0], radius)

        traj_ind_neigh, time_ind_neigh = np.divmod(ind_neighbors, dataset.u.shape[1])
        # Discard neighbors at the end of trajectories (cannot evolve them in time)
        mask_rollout_inrange = time_ind_neigh + rollout < dataset.u.shape[1]

        # Handle insufficient neighbors
        if np.sum(mask_rollout_inrange) < num_neighbor_threshold:
            maps.append(np.zeros_like(us, shape=(rollout, dim_project, dim)))
            continue

        # Find time evolution of the radius r neighborhood
        traj_ind_neigh: Int[np.ndarray, " neighbors"] = traj_ind_neigh[
            mask_rollout_inrange
        ]
        time_ind_neigh: Int[np.ndarray, " neighbors"] = time_ind_neigh[
            mask_rollout_inrange
        ]

        u_neighs: Float[np.ndarray, "neighbors rollout+1 dim"] = np.stack(
            [
                dataset.u[i, j : j + rollout + 1]
                for i, j in zip(traj_ind_neigh, time_ind_neigh)
            ],
            axis=0,
        )
        projectors: Float[np.ndarray, "rollout+1 dim dim_proj"] = tangent_projectors[
            traj_ind, time_ind : time_ind + rollout + 1
        ]

        dus = u_neighs - us

        # Map time evolution of the neighborhood to respective tangent planes
        dus_proj: Float[np.ndarray, "neighbors rollout+1 dim_proj"] = np.einsum(
            "abi,bij->abj", dus, projectors
        )
        map_i: Float[np.ndarray, "rollout dim_proj dim_proj"] = np.stack(
            [
                least_squares(dus_proj[:, 0], dus_proj[:, i])
                for i in range(1, rollout + 1)
            ],
            axis=0,
        )
        map_i: Float[np.ndarray, "rollout dim_proj dim"] = map_i @ rearrange(
            projectors[1:], "batch dim dim_proj -> batch dim_proj dim"
        )
        maps.append(map_i)
    maps = rearrange(
        np.stack(maps, axis=0),
        "(trajs time) roll dim_proj dim -> trajs time roll dim_proj dim",
        trajs=dataset.u.shape[0],
    )
    return tangent_projectors[:, :-rollout], maps, scores[:, :-rollout]


def estimate_pushforward_matrices_old(
    dataset: TimeSeriesDataset,
    radius: float,
    dim_project: int,
    num_neighbor_threshold: int,
):
    maps1, maps2, scores = [], [], []
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
            scores.append([1.0, 1.0])
        else:
            du0: Float[np.ndarray, "neighbors dim"] = dataset.u[*neigh_inds0] - u0
            du0_next = dataset.u[neigh_inds0[0], neigh_inds0[1] + 1] - u1

            proj0: Float[np.ndarray, "dim dim_proj"]
            proj0, score0 = get_tangent_space_projector(du0, dim_project)
            proj1: Float[np.ndarray, "dim dim_proj"]
            proj1, score1 = get_tangent_space_projector(
                dataset.u[*neigh_inds1] - u1, dim_project
            )

            du0_proj = du0 @ proj0
            du0_next_proj = du0_next @ proj1
            A: Float[np.ndarray, "dim_proj dim_proj"] = least_squares(
                du0_proj, du0_next_proj
            )
            maps1.append(proj0)
            maps2.append(A @ proj1.T)
            scores.append([score0, score1])

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
    scores = rearrange(
        np.asarray(scores),
        "(trajs time) d -> trajs time d",
        trajs=dataset.u.shape[0],
    )

    return maps1, maps2, scores
