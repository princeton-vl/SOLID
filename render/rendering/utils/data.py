import blenderproc as bproc
import bpy
import io
import os
import json
import numpy as np
import cv2

from mathutils import Matrix
from blenderproc.python.writer.CocoWriterUtility import CocoWriterUtility


def load_texture_paths(obj, texture_dir, **kwargs):
    color_path = os.path.join(texture_dir, obj["texture"], obj["diff"]) if "diff" in obj else None
    disp_path  = os.path.join(texture_dir, obj["texture"], obj["disp"]) if "disp" in obj else None
    rough_path = os.path.join(texture_dir, obj["texture"], obj["rough"]) if "rough" in obj else None
    return color_path, disp_path, rough_path


def load_paths(
    obj,
    model_dir="",
    texture_dir="",
    **kwargs
):
    model_path = os.path.join(model_dir, obj["model"])
    color_path, disp_path, rough_path = load_texture_paths(obj, texture_dir, **kwargs)
    return model_path, color_path, disp_path, rough_path


def sample_objects(objects, num_objects):
    rand_idxs = np.random.randint(len(objects), size=num_objects)
    rand_objs = [objects[i] for i in rand_idxs]
    return rand_objs


def read_textures(texture_json):
    if texture_json is None:
        return None

    with open(texture_json, "r") as f:
        textures = json.load(f)

    for key in textures:
        textures[key]["texture"] = key
    return list(textures.values())


def read_objects(json_path):
    with open(json_path, "r") as f:
        objects = json.load(f)
    for i, object in enumerate(objects):
        object["category_id"] = i + 1
    return objects


def buffer_encode_png(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    return np.frombuffer(buffer.read(), dtype=np.uint8)


def masks_to_bboxes(
    segmaps,
    attribute_maps,
    threshold=0,
    return_masks=False,
):
    bboxes = {}
    masks  = {}
    for attribute_map in attribute_maps:
        if attribute_map["category_id"] == 0:
            continue

        mask = (segmaps == attribute_map["idx"])
        if mask.sum() < threshold:
            continue
        bbox = CocoWriterUtility.bbox_from_binary_mask(mask)

        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        bbox.append(attribute_map["category_id"] - 1)

        name = attribute_map["name"]
        bboxes[name] = bbox
        masks[name]  = mask
    if return_masks:
        return bboxes, masks
    return bboxes


def get_mask_bbox_ratios(bboxes, masks):
    return {
        key: masks[key].sum() / ((
            bboxes[key][3] - bboxes[key][1]
        ) * (
            bboxes[key][2] - bboxes[key][0]
        )) for key in bboxes
    }


def hide_render(mesh, hide, frame):
    mesh.blender_obj.hide_render = hide
    mesh.blender_obj.keyframe_insert("hide_render", frame=frame)
    for fcurve in mesh.blender_obj.animation_data.action.fcurves:
        kf = fcurve.keyframe_points[-1]
        kf.interpolation = "CONSTANT"


def set_visibilities(objects, scene_objs):  #, segments):
    objects    = [obj for obj in objects if obj.is_valid()]
    scene_objs = [obj for obj in scene_objs if obj.is_valid()]
    frame_end  = bpy.context.scene.frame_end

    for i in range(len(objects)):
        hide_render(objects[i], False, frame_end - 1)

    for scene_obj in scene_objs:
        hide_render(scene_obj, False, frame_end - 1)
        hide_render(scene_obj, True, frame_end)

    for i in range(len(objects)):
        for j in range(len(objects)):
            hide_render(objects[j], i != j, frame_end)
        frame_end += 1
    bpy.context.scene.frame_end = frame_end


def get_visibilities(objects, segments):
    names = [obj.get_name() for obj in objects if obj.is_valid()]
    num_pixels = {}
    for i in range(1, len(segments["instance_segmaps"])):
        seg_maps = segments["instance_segmaps"][i]
        for att_map in segments["instance_attribute_maps"][i]:
            att_name = att_map["name"]
            if att_name in names:
                num_pixels[att_name] = (seg_maps == att_map["idx"]).sum()

    seg_maps = segments["instance_segmaps"][0]
    for att_map in segments["instance_attribute_maps"][0]:
        att_name = att_map["name"]
        if att_name not in num_pixels:
            continue

        num_pixels[att_name] = (
            seg_maps == att_map["idx"]
        ).sum() / num_pixels[att_name]
    return num_pixels


def get_angles(objects, camera_pose):
    objects = [obj for obj in objects if obj.is_valid()]
    inv_cam = np.linalg.inv(camera_pose)

    angles = {}
    for i in range(len(objects)):
        name  = objects[i].get_name()
        angle = inv_cam @ objects[i].get_local2world_mat()
        angle = Matrix(angle).to_euler()

        angles[name] = np.array((
            angle.x,
            angle.y,
            angle.z,
        ), dtype=np.float32)
    return angles


def render_images_and_stats(objects, scene_objs, camera_pose, timers):
    with timers["image"]:
        frame_end = bpy.context.scene.frame_end
        bpy.context.scene.frame_end = 1
        img_data = bproc.renderer.render()

    with timers["segmap"]:
        bpy.context.scene.frame_end = frame_end
        seg_data = bproc.renderer.render_segmap(
            map_by=[
                "instance",
                "class",
                "name",
            ],
            render_colorspace_size_per_dimension=4,
        )

    with timers["masks_to_bboxes"]:
        bboxes, masks = masks_to_bboxes(
            seg_data["instance_segmaps"][0],
            seg_data["instance_attribute_maps"][0],
            return_masks=True,
        )

    with timers["ratios"]:
        ratios = get_mask_bbox_ratios(bboxes, masks)
    with timers["angles"]:
        angles = get_angles(objects, camera_pose)
    with timers["visibilities"]:
        visibilities = get_visibilities(objects, seg_data)

    names = list(bboxes.keys())
    names = [name for name in names if name in angles]
    if not names:
        raise Exception("Empty Scene")

    bboxes = np.array([bboxes[name] for name in names])
    masks  = np.array([masks[name]  for name in names])
    ratios = np.array([ratios[name] for name in names])
    angles = np.array([angles[name] for name in names]).reshape(-1)
    visibilities = np.array([visibilities[name] for name in names])

    masks = np.stack(masks)

    data = {
        "images": img_data["colors"],
        "bboxes": bboxes,
        "masks": masks,
    }

    stats = {
        "visibilities": visibilities,
        "mask_bbox_ratios": ratios,
        "angles": angles,
    }
    return data, stats
