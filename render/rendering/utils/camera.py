import numpy as np
import torch

from mathutils import Vector, Matrix


def get_visible_areas(
    pts,
    camera_pose,
    camera_fov,
    min_distance=0,
):
    fov0, fov1 = camera_fov

    camera_pose = Matrix(camera_pose)
    camera_loc  = Vector((0, 0, 0))
    camera_loc  = camera_pose @ camera_loc
    camera_loc  = torch.tensor(
        camera_loc,
        dtype=torch.float32,
        device=pts.device,
    )

    coor0 = camera_pose @ Vector((-np.tan(fov0 / 2),  np.tan(fov1 / 2), -1))
    coor1 = camera_pose @ Vector(( np.tan(fov0 / 2),  np.tan(fov1 / 2), -1))
    coor2 = camera_pose @ Vector(( np.tan(fov0 / 2), -np.tan(fov1 / 2), -1))
    coor3 = camera_pose @ Vector((-np.tan(fov0 / 2), -np.tan(fov1 / 2), -1))
    coors = torch.tensor(
        [
            coor0,
            coor1,
            coor2,
            coor3,
        ],
        dtype=torch.float32,
        device=pts.device,
    ) - camera_loc.reshape(1, 3)  # 4 x 3

    vecs  = pts - camera_loc.reshape(1, 1, 1, 3)
    dists = torch.pow(vecs[:, :, :, 0:2], 2).sum(dim=3)

    crosses0 = torch.cross(coors, coors[[1, 2, 3, 0]], dim=-1)  # 4 x 3

    coors = coors.reshape(
        4, 1, 1, 1, 3,
    ).expand(
        -1, vecs.shape[0], vecs.shape[1], vecs.shape[2], -1,
    )
    vecs  = vecs.unsqueeze(0).expand(coors.shape[0], -1, -1, -1, -1)

    crosses0 = crosses0.reshape(4, 1, 1, 1, 3)  # 4 x 1 x 1 x 3
    crosses1 = torch.cross(coors, vecs, dim=-1)  # 4 x n x s x 3

    dots  = (crosses0 * crosses1).sum(dim=4)  # 4 x n x s
    keeps = (dots > 0).all(dim=0) & (dists > (min_distance ** 2))  # n x s
    return keeps
