import blenderproc as bproc
import numpy as np
import math
import bpy
import bmesh
import queue
import torch

from typing import List
from mathutils import Vector, Matrix
from blenderproc.python.types.MeshObjectUtility import MeshObject, convert_to_meshes
from blenderproc.python.writer.CocoWriterUtility import CocoWriterUtility

from scipy.spatial import ConvexHull
from rendering.utils.data import load_paths

UP_VECTOR = Vector((0, 0, 1))


def create_plane(scale_x, scale_y):
    plane = bproc.object.create_primitive("PLANE")
    dim_x, dim_y = plane.blender_obj.dimensions[0:2]
    plane.set_scale([scale_x / dim_x, scale_y / dim_y, 1])

    bpy.context.view_layer.objects.active = plane.blender_obj
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.quads_convert_to_tris(
        quad_method="BEAUTY",
        ngon_method="BEAUTY",
    )
    bpy.ops.object.editmode_toggle()

    return plane


def create_cube(scale_x, scale_y, scale_z):
    cube = bproc.object.create_primitive("CUBE")
    dim_x, dim_y, dim_z = cube.blender_obj.dimensions
    cube.set_scale([scale_x / dim_x, scale_y / dim_y, scale_z / dim_z])
    return cube


def rotate_x(loc, angle):
    loc = np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle),  np.cos(angle)],
    ]) @ loc.reshape(-1, 1)
    return loc


def rotate_y(loc, angle):
    loc = np.array([
        [np.cos(angle), 0, -np.sin(angle)],
        [0, 1, 0],
        [np.sin(angle), 0, np.cos(angle)],
    ]) @ loc.reshape(-1, 1)
    return loc


def rotate_z(loc, angle):
    loc = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0, 0, 1]
    ]) @ loc.reshape(-1, 1)
    return loc


def segmaps_per_object(use_name=False):
    num_pixels = {}
    objects = bproc.object.get_all_mesh_objects()
    for i in range(len(objects)):
        for j, obj in enumerate(objects):
            if obj.is_valid():
                obj.hide(i != j)

        seg_data = bproc.renderer.render_segmap(map_by=["instance", "class", "name"])
        segmaps  = seg_data["instance_segmaps"][0]
        if len(seg_data["instance_attribute_maps"][0]) > 1:
            attmaps = seg_data["instance_attribute_maps"][0][1]
            attkey  = attmaps["idx"] if not use_name else attmaps["name"]
            num_pixels[attkey] = (segmaps == attmaps["idx"]).sum()
        objects = bproc.object.get_all_mesh_objects()

    for obj in objects:
        obj.hide(False)
    return num_pixels


def hide_occluded_objects(seg_data, ratio=0.1, num_pixels=None):
    num_pixels = segmaps_per_object() if num_pixels is None else num_pixels

    objects = bproc.object.get_all_mesh_objects()
    segmaps = seg_data["instance_segmaps"][0]
    attmaps = seg_data["instance_attribute_maps"][0]
    for attmap in attmaps:
        if attmap["category_id"] == 0:
            continue

        num_pixel  = num_pixels[attmap["idx"]]
        mask_pixel = (segmaps == attmap["idx"]).sum()

        if mask_pixel / num_pixel < 0.5:
            obj = bproc.filter.one_by_attr(objects, "name", attmap["name"])
            obj.hide()

    data = bproc.renderer.render()
    seg_data = bproc.renderer.render_segmap(map_by=["instance", "class", "name"])
    return data, seg_data


def hide_slender_objects(seg_data, ratio=0.1):
    objects = bproc.object.get_all_mesh_objects()
    segmaps = seg_data["instance_segmaps"][0]
    attmaps = seg_data["instance_attribute_maps"][0]
    for attmap in attmaps:
        if attmap["category_id"] == 0:
            continue

        mask = (segmaps == attmap["idx"])
        bbox = CocoWriterUtility.bbox_from_binary_mask(mask)

        if min(bbox[2] / bbox[3], bbox[3] / bbox[2]) < 0.1:
            obj = bproc.filter.one_by_attr(objects, "name", attmap["name"])
            obj.hide()

    data = bproc.renderer.render()
    seg_data = bproc.renderer.render_segmap(map_by=["instance", "class", "name"])
    return data, seg_data


def load_object(
    obj_id,
    obj,
    no_remap=False,
    tf=None,
    **kwargs
):
    model_path, color_path, disp_path, rough_path = load_paths(obj, **kwargs)
    category_id = obj["category_id"]

    if model_path.endswith("stl"):
        model = load_stl(model_path)[0]
        model.clear_materials()

        material = create_material(f"{obj_id}/material", color_path, disp_path, rough_path)
        model.add_material(material)
    elif model_path.endswith("obj"):
        model = bproc.loader.load_obj(model_path)[0]
    elif model_path.endswith("glb") or model_path.endswith("gltf"):
        model = load_gltf(model_path)[0]
    else:
        raise ValueError("unknown file")

    set_origin(model)

    model.set_name(str(obj_id))
    model.set_cp("category_id", category_id)
    model.set_cp("model_path", model_path)
    return model


def setup_object(
    model,
    min_scale=1.0,
    max_scale=1.0,
    min_thetax=np.pi / 6,
    max_thetax=np.pi / 6,
    min_thetay=0,
    max_thetay=0,
    min_thetaz=0,
    max_thetaz=0,
    scale_set="all",
    **kwargs,
):
    thetax, thetay, thetaz = np.random.uniform(
        [min_thetax, min_thetay, min_thetaz],
        [max_thetax, max_thetay, max_thetaz],
    )

    model.blender_obj.rotation_mode = "XYZ"
    model.set_rotation_euler([thetax, thetay, thetaz])

    scales = {
        "all": [
            [min_scale, max_scale],
        ],
        "small": [
            [0.1, 1.0],
            [1.0, 2.0],
            [2.0, 3.0],
        ],
    }[scale_set]

    scales_probs = {
        "all": [1.0],
        "small": [0.7, 0.1, 0.2],
    }[scale_set]

    scale_indices = list(range(len(scales)))
    scale_index   = np.random.choice(
        scale_indices,
        p=scales_probs,
    )
    min_scale, max_scale = scales[scale_index]

    scale = np.random.uniform(min_scale, max_scale)
    rescale(model, scale)
    model.set_cp("scale", scale)
    return model


def get_rows_and_cols(num_objs):
    num_rows = int(math.sqrt(num_objs))
    num_cols = math.ceil(num_objs / num_rows)
    return num_rows, num_cols


def position_camera(init_location, num_rows, num_cols, tile_size, camera_y):
    init_x = init_location[0]
    init_z = init_location[2]

    x = (2 * init_x + ((num_cols - 1) * tile_size)) / 2
    z = (2 * init_z + ((num_rows - 1) * tile_size)) / 2

    # Create a point light next to it
    light = bproc.types.Light()
    light.set_location([x, camera_y, z])
    # light.set_location([x, -6, z]) # for Thingi10k
    light.set_energy(1000)

    cam_pose = bproc.math.build_transformation_mat(
        [x, camera_y, z], [-np.pi / 2, np.pi, 0]
        # [x, -6, z], [np.pi / 2, 0, 0] # for Thingi10k
    )
    bproc.camera.add_camera_pose(cam_pose)


def rescale(obj, max_size):
    scale = max_size / max(obj.blender_obj.dimensions)
    obj.set_scale([scale, scale, scale])


def remap_uv(obj):
    bpy.context.view_layer.objects.active = obj.blender_obj
    bpy.ops.object.editmode_toggle()
    bpy.ops.uv.cube_project()
    bpy.ops.object.editmode_toggle()


def set_origin(obj):
    bpy.context.view_layer.objects.active = obj.blender_obj
    bpy.ops.object.origin_set(center='BOUNDS')


def load_stl(filepath: str, **kwargs) -> List[MeshObject]:
    previously_selected_objects = bpy.context.selected_objects

    bpy.ops.import_mesh.stl(filepath=filepath, **kwargs)
    return convert_to_meshes([
        obj for obj in bpy.context.selected_objects
        if obj not in previously_selected_objects
    ])


def load_gltf(filepath: str, **kwargs) -> List[MeshObject]:
    previously_selected_objects = bpy.context.selected_objects

    bpy.ops.import_scene.gltf(filepath=filepath, **kwargs)
    if filepath.endswith("gltf"):
        bpy.ops.object.join()

    return convert_to_meshes([
        obj for obj in bpy.context.selected_objects
        if obj not in previously_selected_objects
    ])


def create_material(name, color_path, disp_path, rough_path):
    material = bproc.material.create(name)
    setup_material(material, color_path, disp_path, rough_path)
    return material


def setup_material(material, color_path, disp_path, rough_path):
    mat_nodes = material.nodes
    mat_links = material.links
    bsdf_node = material.get_the_one_node_with_type("BsdfPrincipled")
    outp_node = material.get_the_one_node_with_type("OutputMaterial")

    color_node = bproc.material.add_base_color(
        mat_nodes, mat_links, color_path, bsdf_node
    ) if color_path else None
    disp_node  = bproc.material.add_displacement(
        mat_nodes, mat_links, disp_path, outp_node
    ) if disp_path else None
    rough_node = bproc.material.add_roughness(
        mat_nodes, mat_links, rough_path, bsdf_node
    ) if rough_path else None

    text_nodes = [n for n in [color_node, disp_node, rough_node] if n]
    bproc.material.connect_uv_maps(mat_nodes, mat_links, text_nodes)


def calc_area(verts):
    v = verts[1] - verts[0]
    w = verts[2] - verts[0]
    c = v.cross(w)
    return c.length / 2


def look_for_faces(face, faces, om, use_abs=False):
    results = []

    q = queue.Queue()
    q.put(face)
    while not q.empty():
        face  = q.get()
        verts = [om @ v.co for v in face.verts]

        v = verts[1] - verts[0]
        w = verts[2] - verts[0]

        normal = v.cross(w).normalized()
        cosine = UP_VECTOR.dot(normal)  # TODO: Check if abs is needed

        if use_abs:
            cosine = abs(cosine)

        if cosine < 1 - 1e-1 or cosine > 1 + 1e-1:
            continue

        results.append(verts)
        for edge in face.edges:
            for face in edge.link_faces:
                if face.index in faces:
                    faces.pop(face.index)
                    q.put(face)
    return results


def look_for_surface(obj, use_gltf=False, use_abs=False):
    model_path = obj.get_cp("model_path") if obj.has_cp("model_path") else None

    om  = Matrix(obj.get_local2world_mat())
    obj = obj.blender_obj

    if use_gltf and model_path is not None:
        gltf_obj = load_gltf(model_path)[0]
        obj = gltf_obj.blender_obj

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.editmode_toggle()

    me = obj.data
    bm = bmesh.from_edit_mesh(me)

    faces = {f.index: f for f in bm.faces}
    all_surfaces = []
    while faces:
        face = faces.popitem()[1]
        new_surface = look_for_faces(face, faces, om, use_abs=use_abs)
        if new_surface:
            all_surfaces.append(new_surface)

    def highest_pt(surface):
        max_z = None
        for face in surface:
            for vert in face:
                max_z = vert[2] if max_z is None else max(max_z, vert[2])
        return max_z

    all_surfaces = sorted(
        all_surfaces,
        key=highest_pt,
        reverse=True,
    )

    all_tris = []
    for i, surface in enumerate(all_surfaces):
        area = sum([calc_area(tri) for tri in surface])
        if area < 0.1:
            continue
        all_tris.append(np.array(surface))

    bpy.ops.object.editmode_toggle()

    if use_gltf:
        gltf_obj.delete()
    return all_tris


def same_side(p1, p2, a, b):
    # p1: p x 2
    # p2: n x 2
    # a:  n x 2
    # b:  n x 2
    # res: p x n
    p = p1.shape[0]

    vs = (b - a)[None, :, :]  # 1 x n x 2
    ws = p1[:, None, :] - a[None, :, :]  # p x n x 2
    us = (p2 - a)[None, :, :]  # 1 x n x 2

    cp1 = torch.cross(vs.expand(p, -1, -1), ws, dim=-1)  # p x n x 2
    cp2 = torch.cross(vs, us, dim=-1)  # 1 x n x 2
    dts = (cp1 * cp2).sum(dim=-1)
    return dts >= 0


def points_in_tris(pts, tris, return_indices=False):
    insides = \
        same_side(pts, tris[:, 0, :], tris[:, 1, :], tris[:, 2, :]) & \
        same_side(pts, tris[:, 1, :], tris[:, 0, :], tris[:, 2, :]) & \
        same_side(pts, tris[:, 2, :], tris[:, 0, :], tris[:, 1, :])  # p x n
    any_insides = insides.any(dim=1)  # p

    if return_indices:
        surfaces = torch.where(any_insides, insides.int().argmax(dim=1), -1)
        return any_insides, surfaces
    return any_insides
