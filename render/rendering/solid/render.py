import blenderproc as bproc
import bmesh
import bpy
import torch
import torch.nn as nn
import numpy as np
import gc
import os

from tabulate import tabulate
from mathutils import Vector, Matrix
from collections import OrderedDict
from fft_conv_pytorch import fft_conv

from rendering.init import init
from rendering.utils.time import print_timers, TimerNode
from rendering.utils.data import render_images_and_stats, set_visibilities
from rendering.utils.light import three_points_lighting
from rendering.utils.camera import get_visible_areas
from rendering.utils.object import load_object, setup_object
from rendering.utils.object import create_plane, create_cube
from rendering.utils.object import look_for_surface, points_in_tris
from rendering.utils.scenenet import load_scenenet, get_scenenet_ids
from rendering.utils.argparse import ConvertDegreeToRadian, add_default_arguments, add_render_func

TIMERS = TimerNode()


def print_counts(counts):
    counts = [(key, ) + value for key, value in counts.items()]
    print(tabulate(
        counts,
        headers=["Section", "Before", "After"],
        tablefmt="pretty",
        colalign=("left", "right", "right"),
    ))


def points_in_bbox(points, cube3d):
    """
    cube3d  =  numpy array of the shape (8,3) with coordinates in the clockwise order. first the bottom plane is considered then the top one.
    points = array of points with shape (N, 3).

    Returns the indices of the points array which are outside the cube3d
    """
    t1, t2, t3, t4, b1, b2, b3, b4 = cube3d

    dir1  = (t1 - b1)
    size1 = torch.linalg.norm(dir1)
    dir1  = dir1 / size1

    dir2  = (b2 - b1)
    size2 = torch.linalg.norm(dir2)
    dir2  = dir2 / size2

    dir3  = (b4 - b1)
    size3 = torch.linalg.norm(dir3)
    dir3  = dir3 / size3

    cube3d_center = (t1 + b3) / 2.0

    dir_vec = points - cube3d_center

    res1 = (dir_vec * dir1.reshape(1, 1, 1, 3)).sum(dim=3)
    res1 = torch.abs(res1 * 2) > size1
    res2 = (dir_vec * dir2.reshape(1, 1, 1, 3)).sum(dim=3)
    res2 = torch.abs(res2 * 2) > size2
    res3 = (dir_vec * dir3.reshape(1, 1, 1, 3)).sum(dim=3)
    res3 = torch.abs(res3 * 2) > size3
    return ~(res1 | res2 | res3)


def generate_3D_kernels(bbox, strides, device):
    bbox = torch.tensor(
        bbox,
        device=device,
        dtype=torch.float32,
    )

    strides = torch.tensor(
        strides,
        device=device,
        dtype=torch.float32,
    )

    bbox_ctr = (bbox[0] + bbox[6]) / 2
    bbox -= bbox_ctr

    min_bbox = bbox.min(dim=0)[0]
    max_bbox = bbox.max(dim=0)[0]

    kernel_sizes = (max_bbox - min_bbox) / strides
    kernel_sizes = kernel_sizes.int() + 1

    zs, ys, xs = torch.meshgrid(
        torch.linspace(
            min_bbox[2],
            max_bbox[2],
            steps=kernel_sizes[2],
            device=device,
            dtype=torch.float32,
        ),
        torch.linspace(
            min_bbox[1],
            max_bbox[1],
            steps=kernel_sizes[1],
            device=device,
            dtype=torch.float32,
        ),
        torch.linspace(
            min_bbox[0],
            max_bbox[0],
            steps=kernel_sizes[0],
            device=device,
            dtype=torch.float32,
        ),
        indexing="ij",
    )
    kernel_coors = torch.stack((xs, ys, zs), dim=3)

    keeps  = points_in_bbox(kernel_coors, bbox)
    kernel = torch.zeros(
        (
            kernel_sizes[2].item(),
            kernel_sizes[1].item(),
            kernel_sizes[0].item(),
        ),
        device=device,
        dtype=torch.float32,
    )

    kernel[keeps] = 1.0
    return kernel


def valid_locations(bbox, valids, strides):
    kernel  = generate_3D_kernels(bbox, strides, device=valids.device)
    kernel_size = kernel.shape

    kernel = kernel[None][None]
    num_px = kernel.sum().item()

    padding = [0, 0] * len(kernel_size)
    for k, i in zip(kernel_size, range(len(kernel_size) - 1, -1, -1)):
        total_padding = (k - 1)
        left_padding  = total_padding // 2
        padding[2 * i]     = left_padding
        padding[2 * i + 1] = (total_padding - left_padding)

    valids = valids[None][None].float()
    valids = nn.functional.pad(valids, padding)
    valids = fft_conv(valids, kernel)[0, 0]
    valids = (valids == num_px)
    return valids


def get_axes(bbox_normals, triangle_edges):
    num_triangles = triangle_edges.shape[0]
    bbox_normals = bbox_normals.unsqueeze(0).unsqueeze(2)
    bbox_normals = bbox_normals.expand(num_triangles, -1, 3, -1)
    bbox_normals = bbox_normals.reshape(num_triangles, 9, 3)
    triangle_edges = triangle_edges.unsqueeze(1)
    triangle_edges = triangle_edges.expand(-1, 3, -1, -1)
    triangle_edges = triangle_edges.reshape(num_triangles, 9, 3)

    return torch.cross(
        bbox_normals,
        triangle_edges,
        dim=-1,
    )


def check_intersects(bboxes, triangles, timers):
    # https://gdbooks.gitbooks.io/3dcollisions/content/Chapter4/aabb-triangle.html
    num_triangles = triangles.shape[0]
    batch_size    = 64

    with timers["centers_extents"]:
        centers = (
            bboxes.min(dim=1)[0] +
            bboxes.max(dim=1)[0]
        ) / 2

        extents = (
            bboxes.max(dim=1)[0] -
            bboxes.min(dim=1)[0]
        ) / 2

    with timers["edges_normals"]:
        triangle_edges = torch.diff(
            triangles,
            dim=1,
            append=triangles[:, 0:1, :],
        )
        triangle_normals = torch.cross(
            triangle_edges[:, 0, :],
            triangle_edges[:, 1, :],
            dim=-1,
        )
        triangle_normals = triangle_normals.unsqueeze(1)

        bbox_normals = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            device=bboxes.device,
            dtype=torch.float32,
        )

        normals = torch.cat((
            triangle_normals,
            bbox_normals.unsqueeze(0).expand(num_triangles, -1, -1),
        ), dim=1)

    with timers["get_axes"]:
        axes = get_axes(bbox_normals, triangle_edges)
        axes = torch.cat((
            axes,
            normals,
        ), dim=1)
        axes = axes.unsqueeze(0).unsqueeze(3)

    with timers["proj_bbox"]:
        normal_dots = bbox_normals.unsqueeze(0).unsqueeze(1) * axes[0]
        normal_dots = torch.abs(normal_dots.sum(dim=-1))
        normal_dots = normal_dots.unsqueeze(0)

    num_batches = (num_triangles + batch_size - 1) // batch_size
    extents = extents.unsqueeze(1).unsqueeze(2)

    results = []
    for i in range(num_batches):
        sind = i * batch_size
        eind = min(sind + batch_size, num_triangles)
        
        with timers["proj_tris"]:
            tri_axes = triangles[sind:eind].unsqueeze(1) * axes[0, sind:eind]
            tri_axes = tri_axes.sum(dim=-1)

            ctr_axes = centers.unsqueeze(1).unsqueeze(2) * axes[:, sind:eind, :, 0, :]
            ctr_axes = ctr_axes.sum(dim=-1)

            dots = tri_axes.unsqueeze(0) - ctr_axes.unsqueeze(3)

        with timers["test"]:
            rs = normal_dots[:, sind:eind, :, :] * extents
            rs = rs.sum(dim=-1)

            max_dots = dots.max(dim=-1)[0]
            min_dots = dots.min(dim=-1)[0]
            max_dots = torch.maximum(-max_dots, min_dots)

            not_intersects = (max_dots > rs)
            not_intersects = not_intersects.any(dim=-1)
        results.append(not_intersects)
    results = torch.cat(results, dim=1)
    return ~results


def filter_coors(func):
    def filter_func(
        keeps,
        counts,
        coors,
        *args,
        **kwargs
    ):
        name  = func.__name__
        coors = coors[keeps]
        counts[name] = None

        before = keeps.sum()
        if before > 0:
            ks = func(coors, *args, **kwargs)
            keeps[keeps.clone()] = ks

        after = keeps.sum()
        counts[name] = (before, after)
    return filter_func


def get_camera_pose(
    coors,
    valids,
    *,
    min_y,
    max_y,
    min_z,
    max_z,
    look_min_x,
    look_min_y,
    look_min_z,
    look_max_x,
    look_max_y,
    look_max_z,
    bvh_tree=None,
    min_space,
    **kwargs,
):
    any_valids = valids.any(dim=0)
    valids_ind = valids.int().argmax(dim=0)

    xs, ys = torch.meshgrid(
        torch.arange(valids.shape[2], device=coors.device),
        torch.arange(valids.shape[1], device=coors.device),
        indexing="xy",
    )
    floors = coors[valids_ind, ys, xs][any_valids]

    xys   = floors[:, 0:2]
    dists = torch.sqrt((xys ** 2).sum(axis=1))
    keeps = ((dists >= min_y) & (dists <= max_y))
    keeps = np.where(keeps.cpu().numpy())[0]

    found = True if bvh_tree is None else False

    while True:
        rand_ind = np.random.randint(keeps.shape[0])
        rand_ind = keeps[rand_ind]

        x, y, z = floors[rand_ind].cpu().numpy()

        min_z = max(min_z, z + 1e-3)
        if min_z > max_z:
            max_z = min_z

        z = np.random.uniform(min_z, max_z)
        rand_location = np.array((x, y, z))

        look_at = Vector(
            np.random.uniform([
                look_min_x,
                look_min_y,
                look_min_z + min_z,
            ], [
                look_max_x,
                look_max_y,
                look_max_z + min_z,
            ])
        )

        forward_vec = look_at - Vector(rand_location)
        rand_angle  = bproc.camera.rotation_from_forward_vec(forward_vec)

        if bvh_tree is not None:
            matrix = bproc.math.build_transformation_mat(rand_location, rand_angle)
            found  = bproc.camera.perform_obstacle_in_view_check(
                matrix,
                {
                    "min": min_space,
                },
                bvh_tree,
            )

        if found:
            break
    return rand_location, rand_angle


@filter_coors
def sample_around_objects(
    coors,
    fg_bboxes,
    radius_range=np.inf,
):
    ctrs  = (fg_bboxes[:, 0, :] + fg_bboxes[:, 6, :]) / 2
    radii = torch.sqrt((
        (fg_bboxes - ctrs[:, None, :]) ** 2
    ).sum(dim=2).max(dim=1)[0])

    dists = torch.sqrt((
        (coors[:, None, :] - ctrs[None, :, :]) ** 2
    ).sum(dim=2))
    keeps = (dists < radii[None] + radius_range).any(dim=1)
    return keeps


@filter_coors
def check_collision(
    coors,
    bbox,
    walls,
    matrix,
    timers,
):
    num_coors = coors.shape[0]
    matrix = torch.tensor(
        matrix,
        device=coors.device,
        dtype=torch.float32,
    )

    with timers["rotate"]:
        coors = coors @ matrix.T
        walls = walls @ matrix.T

    bbox = torch.tensor(
        bbox,
        device=coors.device,
        dtype=torch.float32,
    ) @ matrix.T
    with timers["bbox"]:
        bboxes  = bbox.unsqueeze(0).expand(num_coors, -1, -1).clone()
        bboxes += coors.unsqueeze(1)

    with timers["intersect"]:
        keeps = check_intersects(bboxes, walls, timers["intersect"])

    return ~keeps.any(dim=1)


def create_ceiling(plane_scale_x, plane_scale_y, emission_color=None):
    ceiling = create_plane(
        plane_scale_x,
        plane_scale_y
    )
    ceiling.set_location((0, 0, max(plane_scale_x, plane_scale_y)))
    emission_strength = np.random.uniform(0.0, 1.0)
    bproc.lighting.light_surface(
        [ceiling],
        emission_strength=emission_strength,
        emission_color=emission_color,
    )


def get_projection_matrix(device=None):
    camera = bpy.context.scene.camera

    modelview_matrix  = camera.matrix_world.inverted()
    projection_matrix = camera.calc_matrix_camera(
        bpy.data.scenes["Scene"].view_layers["View Layer"].depsgraph,
        x=bpy.context.scene.render.resolution_x,
        y=bpy.context.scene.render.resolution_y,
        scale_x=bpy.context.scene.render.pixel_aspect_x,
        scale_y=bpy.context.scene.render.pixel_aspect_y,
    )

    projection_matrix = projection_matrix @ modelview_matrix
    projection_matrix = torch.tensor(
        projection_matrix,
        device=device,
        dtype=torch.float32,
    )
    return projection_matrix


def get_object_vertices(obj, device=None):
    matrix_world = obj.get_local2world_mat()
    matrix_world = torch.tensor(
        matrix_world,
        device=device,
        dtype=torch.float32,
    )

    vertices = [v.co.to_tuple() + (1, ) for v in obj.blender_obj.data.vertices]
    vertices = torch.tensor(
        vertices,
        device=device,
        dtype=torch.float32,
    ).T
    vertices = matrix_world @ vertices
    return vertices.T


def get_object_faces(obj, device=None):
    matrix_world = obj.get_local2world_mat()
    matrix_world = torch.tensor(
        matrix_world,
        device=device,
        dtype=torch.float32,
    )

    bpy.context.view_layer.objects.active = obj.blender_obj
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.quads_convert_to_tris(
        quad_method="BEAUTY",
        ngon_method="BEAUTY",
    )

    me = obj.blender_obj.data
    bm = bmesh.from_edit_mesh(me)

    faces = [[v.co.to_tuple() + (1, ) for v in face.verts] for face in bm.faces]
    faces = torch.tensor(
        faces,
        device=device,
        dtype=torch.float32,
    )
    num_faces = faces.shape[0]
    faces = faces.reshape(num_faces * 3, 4)
    faces = matrix_world @ faces.T
    faces = faces.T.reshape(num_faces, 3, 4)[:, :, 0:3]
    bpy.ops.object.editmode_toggle()
    return faces


def project_points(proj_matrix, points):
    shape  = list(points.shape)
    points = points.reshape(-1, 4)

    res_x = bpy.context.scene.render.resolution_x
    res_y = bpy.context.scene.render.resolution_y

    proj_points = points @ proj_matrix.T
    proj_points[:, 0] /= proj_points[:, 3]
    proj_points[:, 0] += 1
    proj_points[:, 0] /= 2
    proj_points[:, 0] *= (res_x - 1)

    proj_points[:, 1] /= proj_points[:, 3]
    proj_points[:, 1] -= 1
    proj_points[:, 1] /= 2
    proj_points[:, 1] *= (res_y - 1)
    return proj_points.reshape(*shape)


def within_object(vertices, projs):
    mins = vertices.min(dim=1)[0]
    maxs = vertices.max(dim=1)[0]

    return (
        (projs[:, :, :, 0] >= mins[0]) &
        (projs[:, :, :, 0] <= maxs[0]) &
        (projs[:, :, :, 1] >= mins[1]) &
        (projs[:, :, :, 1] <= maxs[1])
    )


def check_distance(
    coors,
    camera_location,
    min_distance,
    max_distance,
):
    dist = coors[:, :, :, 0:2] - camera_location[0:2].reshape(1, 1, 1, -1)
    dist = torch.norm(dist, dim=3)
    return (dist > min_distance) & (dist < max_distance)


@filter_coors
def scale_distance(
    coors,
    scales,
    dists,
    fg_scale,
    camera_location,
):
    valid_dists = coors - camera_location.reshape(1, -1)
    valid_dists = torch.norm(valid_dists, dim=1)

    sort_inds = torch.argsort(scales)
    scales = scales[sort_inds]
    dists  = dists[sort_inds]

    def pad(array):
        pad_array = torch.zeros(
            (array.shape[0] + 2, ),
            device=array.device,
            dtype=array.dtype,
        )
        pad_array[0]    = -torch.inf
        pad_array[-1]   = torch.inf
        pad_array[1:-1] = array
        return pad_array

    scales = pad(scales)
    dists  = pad(dists)

    keep_inds = (fg_scale < scales)
    max_ind  = torch.argmax(keep_inds.int()) 
    min_ind  = max_ind - 1
    max_dist = dists[max_ind]
    min_dist = dists[min_ind]
    return (valid_dists < max_dist) & (valid_dists > min_dist)


def place_objects(
    coors,
    visibles,
    valids,
    objects,
    num_objects,
    camera_location,
    camera_pose,
    strides,
    timers,
    around_object=1.0,
    occlude_object=0.0,
    radius_range=np.inf,
    obstacles=[],
    max_try=1,
    reuse_object=0.0,
    min_distance=0.0,
    max_distance=torch.inf,
    scale_object=0.0,
    **kwargs,
):
    coors  = coors.clone()
    valids = valids.clone()
    visibles = visibles.clone()

    proj_matrix = get_projection_matrix(device=coors.device)

    camera_pose = torch.tensor(
        camera_pose,
        device=coors.device,
        dtype=coors.dtype,
    )[None]

    fg_objects = []
    fg_bboxes  = torch.zeros(
        (num_objects, 8, 3),
        dtype=torch.float32,
        device=coors.device,
    )
    fg_scales  = torch.zeros(
        (num_objects, ),
        dtype=torch.float32,
        device=coors.device,
    )
    fg_dists   = torch.zeros(
        (num_objects, ),
        dtype=torch.float32,
        device=coors.device,
    )

    inds = valids.nonzero()
    mins = inds.min(dim=0)[0]
    maxs = inds.max(dim=0)[0]

    coors    = coors[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
    valids   = valids[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
    visibles = visibles[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]

    fprojs = torch.zeros_like(valids)
    projs  = project_points(proj_matrix, torch.cat(
        (
            coors,
            torch.ones(
                (coors.shape[0], coors.shape[1], coors.shape[2], 1),
                device=coors.device,
                dtype=coors.dtype,
            ),
        ),
        dim=3,
    ))

    camera_location = torch.tensor(
        camera_location,
        device=coors.device,
        dtype=torch.float32,
    )

    valids &= check_distance(
        coors,
        camera_location,
        min_distance,
        max_distance,
    )

    prev_obj = None
    for i in range(num_objects):
        with timers["load"]:
            if prev_obj is None or np.random.uniform() >= reuse_object:
                fg_obj = np.random.choice(objects)
            else:
                fg_obj = prev_obj
            prev_obj = fg_obj
            fg_obj = load_object(i, fg_obj, **kwargs)
        bpy.ops.object.select_all(action="DESELECT")

        for j in range(max_try):
            counts = OrderedDict()

            fg_obj = setup_object(fg_obj, **kwargs)
            fg_obj.set_location((0, 0, 0))

            fg_bbox   = fg_obj.get_bound_box()
            fg_matrix = Matrix(fg_obj.get_local2world_mat())
            fg_inv    = np.array(fg_matrix.to_quaternion().inverted().to_matrix())

            with timers["valid"]:
                before = valids.sum()
                keeps  = valid_locations(fg_bbox, valids, strides) & valids
                after  = keeps.sum()
                counts["valid"] = (before, after)

                before = keeps.sum()
                keeps &= visibles
                after  = keeps.sum()
                counts["visibles"] = (before, after)
            
            num_objects = len(fg_objects)
            with timers["around"]:
                if num_objects > 0 and np.random.uniform() <= around_object:
                    sample_around_objects(
                        keeps,
                        counts,
                        coors,
                        fg_bboxes[:num_objects],
                        radius_range=radius_range,
                    )

            with timers["occlude"]:
                if num_objects > 0 and np.random.uniform() <= occlude_object:
                    before = keeps.sum()
                    keeps &= fprojs
                    after  = keeps.sum()
                    counts["occlude"] = (before, after)

                    with timers["scale_distance"]:
                        if np.random.uniform() <= scale_object:
                            scale_distance(
                                keeps,
                                counts,
                                coors,
                                fg_scales[:num_objects],
                                fg_dists[:num_objects],
                                fg_obj.get_cp("scale"),
                                camera_location,
                            )

            with timers["collision"]:
                if obstacles is not None:
                    check_collision(
                        keeps,
                        counts,
                        coors,
                        fg_bbox,
                        obstacles,
                        fg_inv,
                        timers["collision"],
                    )

            print_counts(counts)
            if keeps.sum().item() > 0:
                break

        if keeps.sum().item() == 0:
            fg_obj.delete()
            continue

        indices = torch.nonzero(keeps).cpu().numpy()
        ind = np.random.permutation(indices.shape[0])[0]
        ind = indices[ind]

        fg_obj.set_location((
            coors[ind[0], ind[1], ind[2], 0].item(),
            coors[ind[0], ind[1], ind[2], 1].item(),
            coors[ind[0], ind[1], ind[2], 2].item(),
        ), frame=0)

        fg_bbox = fg_obj.get_bound_box()
        fg_bbox = torch.tensor(
            fg_bbox,
            device=coors.device,
            dtype=torch.float32,
        )

        fg_dist = (fg_bbox[0] + fg_bbox[6]) / 2 - camera_location
        fg_dist = torch.norm(fg_dist)

        keeps   = points_in_bbox(coors, fg_bbox)
        valids &= ~keeps

        fg_vertices = get_object_vertices(fg_obj, device=coors.device)
        pj_vertices = project_points(proj_matrix, fg_vertices)

        fprojs |= within_object(pj_vertices, projs)

        fg_bboxes[num_objects] = fg_bbox
        fg_scales[num_objects] = fg_obj.get_cp("scale")
        fg_dists[num_objects]  = fg_dist
        fg_objects.append(fg_obj)

        if valids.sum() == 0:
            break

    del camera_pose
    del fg_bboxes
    del coors
    del fprojs
    del projs
    del keeps
    del pj_vertices
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    return fg_objects


def setup_scenenet(
    plane_scale_x,
    plane_scale_y,
    scenenet_dir,
    scenenet_texture_dir,
    timers,
    **kwargs,
):
    scenenet_ids = get_scenenet_ids()
    scenenet_id  = np.random.choice(scenenet_ids)
    scene = f"{scenenet_id:03d}.glb"
    scene_path = os.path.join(scenenet_dir, scene)
    return load_scenenet(
        scene_path,
        scenenet_texture_dir,
        [
            plane_scale_x,
            plane_scale_y,
            max(plane_scale_x, plane_scale_y),
        ],
        timers,
    ), scene_path


def coors_on_floors(coors, floors):
    valids = torch.ones(
        (coors.shape[0], coors.shape[1], coors.shape[2]),
        device=coors.device,
        dtype=torch.bool,
    )

    min_heights = torch.empty(
        (coors.shape[1], coors.shape[2]),
        device=coors.device,
        dtype=torch.float32,
    ).fill_(-torch.inf)

    proj_coors = coors[0].clone()
    proj_coors[:, :, 2] = 0

    height, width = proj_coors.shape[0:2]

    proj_coors = proj_coors.reshape(height * width, 3)
    for i, floor in enumerate(floors):
        proj_floor = floor.clone()
        proj_floor[:, :, 2] = 0

        valid, index = points_in_tris(
            proj_coors,
            proj_floor,
            return_indices=True,
        )

        floor = floor[index]

        max_floor_z = floor[:, :, 2].max(dim=1)[0]
        max_floor_z = max_floor_z.reshape(height, width)

        valid = valid.reshape(height, width)
        min_heights[valid] = torch.maximum(
            min_heights[valid],
            max_floor_z[valid],
        )

        heights = coors[:, :, :, 2] >= min_heights.unsqueeze(0)
        valids &= heights
    valids &= (min_heights > -torch.inf).unsqueeze(0)
    return valids


def coors_under_ceilings(coors, ceilings):
    valids = torch.ones(
        (coors.shape[0], coors.shape[1], coors.shape[2]),
        device=coors.device,
        dtype=torch.bool,
    )

    max_heights = torch.empty(
        (coors.shape[1], coors.shape[2]),
        device=coors.device,
        dtype=torch.float32,
    ).fill_(torch.inf)

    proj_coors = coors[0].clone()
    proj_coors[:, :, 2] = 0

    height, width = proj_coors.shape[0:2]

    proj_coors = proj_coors.reshape(height * width, 3)
    for ceiling in ceilings:
        proj_ceiling = ceiling.clone()
        proj_ceiling[:, :, 2] = 0

        valid, index = points_in_tris(
            proj_coors,
            proj_ceiling,
            return_indices=True,
        )

        ceiling = ceiling[index]

        max_ceiling_z = ceiling[:, :, 2].max(dim=1, keepdims=True)[0]
        min_ceiling_z = ceiling[:, :, 2].min(dim=1, keepdims=True)[0]
        avg_ceiling_z = (max_ceiling_z + min_ceiling_z) / 2
        avg_ceiling_z = avg_ceiling_z.reshape(height, width)

        valid = valid.reshape(height, width)
        max_heights[valid] = torch.minimum(
            max_heights[valid],
            avg_ceiling_z[valid],
        )

        heights = coors[:, :, :, 2] <= max_heights.unsqueeze(0)
        valids &= heights
    return valids


def generate_coors(
    limits,
    steps=[1, 1, 1],
    device=None,
):
    min_x, max_x = limits[0]
    min_y, max_y = limits[1]
    min_z, max_z = limits[2]

    zs, ys, xs = torch.meshgrid(
        torch.linspace(
            min_z,
            max_z,
            steps=steps[2],
            dtype=torch.float32,
            device=device,
        ),
        torch.linspace(
            min_y,
            max_y,
            steps=steps[1],
            dtype=torch.float32,
            device=device,
        ),
        torch.linspace(
            min_x,
            max_x,
            steps=steps[0],
            dtype=torch.float32,
            device=device,
        ),
        indexing="ij",
    )
    xs = xs.reshape(-1, steps[0])
    ys = ys.reshape(-1, steps[1])
    zs = zs.reshape(-1, steps[2])
    coors = torch.stack((xs, ys, zs), dim=2)
    coors = coors.reshape(steps[2], steps[1], steps[0], 3)
    return coors


def dynamic_sky_lighting():
    bpy.ops.sky.dyn()
    bpy.context.scene.world = bpy.data.worlds.get(
        bpy.context.scene.dynamic_sky_name,
    )


def init_set(
    plane_scale_x,
    plane_scale_y,
    use_scenenet,
    timers,
    device=None,
    **kwargs,
):
    if use_scenenet:
        with timers["setup"]:
            (walls, floors, ceilings, others), scene_path = setup_scenenet(
                plane_scale_x,
                plane_scale_y,
                timers=timers["setup"],
                **kwargs,
            )

        scene_objs = walls + floors + ceilings
        with timers["rigid"]:
            for obj in scene_objs:
                obj.enable_rigidbody(False)

        with timers["bvh_tree"]:
            bvh_tree = bproc.object.create_bvh_tree_multi_objects(walls)
    else:
        plane = create_plane(plane_scale_x, plane_scale_y)
        plane.enable_rigidbody(False)

        floors   = [plane]
        ceilings = []
        walls    = None
        others   = None
        bvh_tree = None
        scene_path = None
        scene_objs = []

    with timers["surface"]:
        floors = [f for floor in floors for f in look_for_surface(floor, use_abs=True)]
        floors = [torch.tensor(
            floor,
            device=device,
            dtype=torch.float32,
        ) for floor in floors]

        ceilings = [c for ceiling in ceilings for c in look_for_surface(ceiling, use_abs=True)]
        ceilings = [torch.tensor(
            ceiling,
            device=device,
            dtype=torch.float32,
        ) for ceiling in ceilings]

        if walls is not None:
            walls = [get_object_faces(wall, device=device) for wall in walls]
            walls = torch.cat(walls, dim=0) if walls else torch.empty(
                (0, 3, 3), device=device, dtype=torch.float32,
            )

        if others is not None:
            others = [get_object_faces(other, device=device) for other in others]
            others = torch.cat(others, dim=0) if others else torch.empty(
                (0, 3, 3), device=device, dtype=torch.float32,
            )
    return floors, walls, ceilings, others, bvh_tree, scene_path, scene_objs


@add_default_arguments("solid")
def add_parser(parser):
    parser.add_argument("--min_y", default=3.6, type=float)
    parser.add_argument("--max_y", default=6.0, type=float)
    parser.add_argument("--min_z", default=0.1, type=float)
    parser.add_argument("--max_z", default=2.0, type=float)
    parser.add_argument("--plane_scale_x", default=12, type=float)
    parser.add_argument("--plane_scale_y", default=12, type=float)
    parser.add_argument("--look_min_x", default=-2.0, type=float)
    parser.add_argument("--look_max_x", default=2.0, type=float)
    parser.add_argument("--look_min_y", default=-2.0, type=float)
    parser.add_argument("--look_max_y", default=2.0, type=float)
    parser.add_argument("--look_min_z", default=0.0, type=float)
    parser.add_argument("--look_max_z", default=2.0, type=float)
    parser.add_argument("--min_size", default=0.01, type=float)
    parser.add_argument("--max_size", default=0.25, type=float)
    parser.add_argument(
        "--min_spread",
        default=np.pi,
        type=float,
        action=ConvertDegreeToRadian
    )
    parser.add_argument(
        "--max_spread",
        default=np.pi,
        type=float,
        action=ConvertDegreeToRadian
    )
    parser.add_argument("--radius_range", default=1.0, type=float)
    parser.add_argument("--min_objects", default=20, type=int)
    parser.add_argument("--max_objects", default=25, type=int)
    parser.add_argument("--around_object", default=1.0, type=float)
    parser.add_argument("--occlude_object", default=0.7, type=float)

    parser.add_argument("--step_size", default=12, type=int)
    parser.add_argument("--use_scenenet", action="store_true")
    parser.add_argument("--min_distance", default=2.0, type=float)
    parser.add_argument("--max_distance", default=torch.inf, type=float)
    parser.add_argument("--min_space", default=4.0, type=float)
    parser.add_argument("--passive_prob", default=0.0, type=float)
    parser.add_argument("--max_try", default=5, type=int)
    parser.add_argument("--reuse_object", default=0.5, type=float)
    parser.add_argument(
        "--custom_fov0",
        default=None,
        type=float,
        action=ConvertDegreeToRadian
    )
    parser.add_argument(
        "--custom_fov1",
        default=None,
        type=float,
        action=ConvertDegreeToRadian
    )
    parser.add_argument("--scale_object", default=0.5, type=float)


@add_render_func("solid")
def render(
    objects,
    min_objects,
    max_objects,
    res_w=320,
    res_h=240,
    min_y=0.6,
    max_y=1.0,
    min_z=1.0,
    max_z=5.0,
    plane_scale_x=6,
    plane_scale_y=6,
    look_min_x=0.0,
    look_max_x=0.0,
    look_min_y=0.0,
    look_max_y=0.0,
    look_min_z=0.0,
    look_max_z=0.0,
    min_size=0.01,
    max_size=0.25,
    min_spread=np.pi,
    max_spread=np.pi,
    radius_range=np.inf,
    around_object=1.0,
    occlude_object=0.0,
    step_size=20,
    min_distance=0,
    max_distance=torch.inf,
    use_scenenet=False,
    min_space=4.0,
    passive_prob=0.0,
    max_try=1,
    reuse_object=0.0,
    custom_fov0=None,
    custom_fov1=None,
    scales=[1],
    scale_object=0.0,
    **kwargs,
):
    with TIMERS:
        if "num_objects" in kwargs:
            del kwargs["num_objects"]

        device = torch.device("cuda") if torch.cuda.is_available() else None

        bproc.init()
        bproc.camera.set_resolution(res_w, res_h)

        num_objects = np.random.randint(
            low=min_objects,
            high=max_objects + 1,
        )

        with TIMERS["init_set"]:
            floors, walls, ceilings, others, bvh_tree, scene_path, scene_objs = init_set(
                plane_scale_x,
                plane_scale_y,
                use_scenenet,
                TIMERS["init_set"],
                device=device,
                **kwargs,
            )
        bpy.ops.object.select_all(action="DESELECT")
        print(f"scene_path: {scene_path}")

        with TIMERS["generate_coors"]:
            limits = (
                (-plane_scale_x / 2, plane_scale_x / 2),
                (-plane_scale_y / 2, plane_scale_y / 2),
                (0, max(plane_scale_x, plane_scale_y)),
            )
            steps   = [int(limit[1] - limit[0]) * step_size for limit in limits]
            strides = [(limit[1] - limit[0]) / (step - 1) for limit, step in zip(limits, steps)]
            coors   = generate_coors(
                limits,
                steps=steps,
                device=device,
            )
            valids = coors_on_floors(coors, floors) & coors_under_ceilings(coors, ceilings)

        with TIMERS["get_camera_pose"]:
            rand_location, rand_angle = get_camera_pose(
                coors,
                valids,
                look_min_x=look_min_x,
                look_min_y=look_min_y,
                look_min_z=look_min_z,
                look_max_x=look_max_x,
                look_max_y=look_max_y,
                look_max_z=look_max_z,
                min_y=min_y,
                max_y=max_y,
                min_z=min_z,
                max_z=max_z,
                bvh_tree=bvh_tree,
                min_space=min_space,
            )
            rand_location = rand_location.reshape(-1)
            camera_fov  = bproc.camera.get_fov()
            camera_pose = bproc.math.build_transformation_mat(
                rand_location,
                rand_angle,
            )

            bproc.camera.add_camera_pose(camera_pose)

        camera_fov = (
            camera_fov[0] + custom_fov0 if custom_fov0 is not None else camera_fov[0],
            camera_fov[1] + custom_fov1 if custom_fov1 is not None else camera_fov[1],
        )
        with TIMERS["get_visible_areas"]:
            visibles = get_visible_areas(
                coors,
                camera_pose,
                camera_fov,
            )

        with TIMERS["place_objects"]:
            obstacles = None
            if walls is not None and others is not None:
                obstacles = torch.cat([walls, others] + floors, dim=0)

            fg_objects = place_objects(
                coors,
                visibles,
                valids,
                objects,
                num_objects,
                camera_location=rand_location,
                camera_pose=camera_pose,
                strides=strides,
                timers=TIMERS["place_objects"],
                around_object=around_object,
                occlude_object=occlude_object,
                radius_range=radius_range,
                obstacles=obstacles,
                max_try=max_try,
                reuse_object=reuse_object,
                min_distance=min_distance,
                max_distance=max_distance,
                scale_object=scale_object,
                **kwargs,
            )

        target_obj = np.random.choice(fg_objects)
        with TIMERS["three_points_lighting"]:
            three_points_lighting(
                coors,
                valids,
                target_obj,
                rand_location,
                min_size=min_size,
                max_size=max_size,
                min_spread=min_spread,
                max_spread=max_spread,
            )

        with TIMERS["create_ceilings"]:
            if not use_scenenet:
                create_cube(
                    plane_scale_x,
                    plane_scale_y,
                    2 * max(plane_scale_x, plane_scale_y)
                )
                create_ceiling(
                    plane_scale_x,
                    plane_scale_y,
                )

        with TIMERS["clean_cache"]:
            del coors
            del visibles
            del valids
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()

        with TIMERS["render"]:
            set_visibilities(fg_objects, scene_objs)

            all_data = []
            for scale in scales:
                bproc.camera.set_resolution(res_w * scale, res_h * scale)
                data = render_images_and_stats(
                    fg_objects,
                    scene_objs,
                    camera_pose,
                    TIMERS["render"],
                )
                all_data.append(data)

    print_timers(TIMERS)
    return all_data
