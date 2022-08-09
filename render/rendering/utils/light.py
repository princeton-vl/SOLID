import blenderproc as bproc
import torch
import numpy as np

from rendering.utils.object import rotate_z


def same_dirs(vecs0, vecs1, vecs2):
    ref_dirs = torch.cross(vecs0, vecs1, dim=-1)  # n x 3

    n = vecs0.shape[0]
    k = vecs2.shape[0]

    vecs0 = vecs0[:, None, :].expand(-1, k, -1)
    vecs2 = vecs2[None, :, :].expand(n, -1, -1)

    vec_dirs = torch.cross(vecs0, vecs2, dim=-1)  # n x k x 3

    dot_prods = ref_dirs[:, None, :] * vec_dirs
    dot_prods = dot_prods.sum(dim=2)  # n x k
    return (dot_prods > 0)


def sample_position(
    keeps,
    floors,
    ceilings,
    min_z,
    max_z,
):
    keeps = torch.nonzero(keeps).reshape(-1)
    if keeps.shape[0] == 0:
        return None

    rand_ind = torch.randint(keeps.shape[0], size=(1,), device=keeps.device)[0]
    rand_ind = keeps[rand_ind]

    floor   = floors[rand_ind]
    ceiling = ceilings[rand_ind]

    location = floor.clone()

    min_z = max(floor[2], min_z)
    max_z = min(ceiling[2], max_z)
    z = (max_z - min_z) * torch.rand((1, ), device=location.device) + min_z

    location[2] = z
    return location


def get_light_position(
    location_ref,
    keeps,
    floors,
    ceilings,
    angle0,
    angle1,
    min_z,
    max_z,
):
    keeps = keeps.clone()
    location_ref0 = rotate_z(location_ref, angle0).reshape(-1)
    location_ref1 = rotate_z(location_ref, angle1).reshape(-1)

    location_ref0 = torch.tensor(
        location_ref0,
        device=floors.device,
        dtype=torch.float32,
    )[None]  # 1 x 3
    location_ref1 = torch.tensor(
        location_ref1,
        device=floors.device,
        dtype=torch.float32,
    )[None]  # 1 x 3

    coors = floors[keeps]  # k x 3
    inds  = torch.nonzero(keeps).reshape(-1)
    keeps[inds] = (
        same_dirs(location_ref0, location_ref1, coors) &
        same_dirs(location_ref1, location_ref0, coors)
    )[0]

    return sample_position(
        keeps,
        floors,
        ceilings,
        min_z,
        max_z,
    )


def three_points_lighting(
    coors,
    valids,
    target_object,
    location_cam,
    min_size=0.01,
    max_size=0.25,
    min_spread=np.pi,
    max_spread=np.pi,
    min_key=150,
    max_key=300,
    min_fill=10,
    max_fill=30,
    min_back=20,
    max_back=60,
    min_key_z=2.0,
    max_key_z=5.0,
    min_fill_z=0.1,
    max_fill_z=1.0,
    min_back_z=6.0,
    max_back_z=10.0,
):
    location_cam = torch.tensor(
        location_cam,
        device=coors.device,
    )

    any_valids   = valids.any(dim=0)
    floor_inds   = valids.int().argmax(dim=0)
    ceiling_inds = valids.shape[0] - torch.flip(valids, dims=(0, )).int().argmax(dim=0) - 1

    ys, xs = torch.meshgrid(
        torch.arange(valids.shape[1], device=coors.device),
        torch.arange(valids.shape[2], device=coors.device),
        indexing="ij",
    )

    floors   = coors[floor_inds, ys, xs][any_valids]
    ceilings = coors[ceiling_inds, ys, xs][any_valids]

    cam_dist = torch.sqrt((location_cam[0:2] ** 2).sum()).item()
    min_dist = 0.8 * cam_dist
    max_dist = 1.0 * cam_dist

    dists = torch.sqrt((floors[:, 0:2] ** 2).sum(dim=1))
    keeps = (dists >= min_dist) & (dists <= max_dist)

    # Sample key light position
    location_key = sample_position(
        keeps,
        floors,
        ceilings,
        min_key_z,
        max_key_z,
    ).cpu().numpy()

    light_key = bproc.types.Light("AREA", name="Key Light")
    light_key.set_location(location_key)
    light_key.set_energy(np.random.uniform(min_key, max_key))
    light_key.blender_obj.data.size   = np.random.uniform(min_size, max_size)
    light_key.blender_obj.data.spread = np.random.uniform(min_spread, max_spread)

    track_key = light_key.blender_obj.constraints.new(type="TRACK_TO")
    track_key.target = target_object.blender_obj
    track_key.track_axis = "TRACK_NEGATIVE_Z"
    track_key.up_axis = "UP_Y"

    # Sample fill light position
    location_fill = get_light_position(
        location_key,
        keeps,
        floors,
        ceilings,
        np.pi / 3,
        -np.pi / 3,
        min_fill_z,
        max_fill_z,
    ).cpu().numpy()

    light_fill = bproc.types.Light("AREA", name="Fill Light")
    light_fill.set_location(location_fill)
    light_fill.set_energy(np.random.uniform(min_fill, max_fill))

    track_fill = light_fill.blender_obj.constraints.new(type="TRACK_TO")
    track_fill.target = target_object.blender_obj
    track_fill.track_axis = "TRACK_NEGATIVE_Z"
    track_fill.up_axis = "UP_Y"

    # Sample back light position
    location_back = get_light_position(
        location_cam.cpu().numpy(),
        keeps,
        floors,
        ceilings,
        np.pi / 6 + np.pi,
        -np.pi / 6 + np.pi,
        min_back_z,
        max_back_z,
    )

    if location_back is not None:
        location_back = location_back.cpu().numpy()

        light_back = bproc.types.Light("AREA", name="Back Light")
        light_back.set_location(location_back)
        light_back.set_energy(np.random.uniform(min_back, max_back))

        track_back = light_back.blender_obj.constraints.new(type="TRACK_TO")
        track_back.target = target_object.blender_obj
        track_back.track_axis = "TRACK_NEGATIVE_Z"
        track_back.up_axis = "UP_Y"
