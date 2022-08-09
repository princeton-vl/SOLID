# ------------------------------------------------------------------------------
# This file contains code from
# BlenderProc (https://github.com/DLR-RM/BlenderProc)
# Licensed under the GPLv3 License
# ------------------------------------------------------------------------------

import blenderproc as bproc
import bpy
import numpy as np
import os
import glob
import random

from typing import List
from rendering.utils.object import set_origin, load_gltf

from blenderproc.python.utility.LabelIdMapping import LabelIdMapping
from blenderproc.python.types.MeshObjectUtility import MeshObject


def get_scenenet_ids():
    scenenet_ids = [
         0,  3,  4,  5,  6,  7,  8,  9, 10, 12,
        13, 15, 17, 18, 25, 26, 27, 28, 29, 30, 
        31, 32, 37, 38, 39, 41, 42, 43, 45, 46,
        50, 51, 52, 53, 54, 56, 57, 58, 59, 60,
    ]
    return scenenet_ids


def filter_objs_by_cat(objs, mapping, cat, id_only=False):
    cid = mapping.id_from_label(cat)
    res = []
    for obj in objs:
        check = (
            (obj.get_cp("category_id") == cid) or
            (not id_only and (cat.lower() in obj.get_name().lower()))
        )
        if check:
            res.append(obj)
    return res


def get_centers(objects):
    bboxes = [obj.get_bound_box() for obj in objects]
    bboxes = np.concatenate(bboxes)
    min_x = bboxes[:, 0].min()
    max_x = bboxes[:, 0].max()
    min_y = bboxes[:, 1].min()
    max_y = bboxes[:, 1].max()

    ctr_x = (min_x + max_x) / 2
    ctr_y = (min_y + max_y) / 2
    return ctr_x, ctr_y


def get_width_height(objects):
    bboxes = [obj.get_bound_box() for obj in objects]
    bboxes = np.concatenate(bboxes)
    min_x = bboxes[:, 0].min()
    max_x = bboxes[:, 0].max()
    min_y = bboxes[:, 1].min()
    max_y = bboxes[:, 1].max()
    min_z = bboxes[:, 2].min()
    max_z = bboxes[:, 2].max()

    width  = max_x - min_x
    height = max_y - min_y
    depth  = max_z - min_z
    return width, height, depth


def set_offsets(objects, offsets):
    for obj in objects:
        location  = obj.get_location()
        location += offsets


def center_scenenet(new_objs):
    bboxes = np.concatenate([f.get_bound_box() for f in new_objs])
    min_x = bboxes[:, 0].min()
    max_x = bboxes[:, 0].max()
    min_y = bboxes[:, 1].min()
    max_y = bboxes[:, 1].max()
    min_z = bboxes[:, 2].min()
    ctr_x = (min_x + max_x) / 2
    ctr_y = (min_y + max_y) / 2
    for obj in new_objs:
        obj.select()
    bpy.context.scene.tool_settings.transform_pivot_point = "BOUNDING_BOX_CENTER"
    bpy.ops.transform.translate(
        value=(-ctr_x, -ctr_y, -min_z),
        orient_type='GLOBAL',
        orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        orient_matrix_type='GLOBAL',
        constraint_axis=(True, False, False),
        mirror=True,
        use_proportional_edit=False,
        proportional_edit_falloff='SMOOTH',
        proportional_size=1,
        use_proportional_connected=False,
        use_proportional_projected=False,
    )
    for obj in new_objs:
        obj.deselect()


def rescale_scenenet(sizes, objs):
    floors   = objs["floors"]
    walls    = objs["walls"]
    ceilings = objs["ceilings"]

    width, height, depth = get_width_height(floors + walls + ceilings)
    scale_x = sizes[0] / width
    scale_y = sizes[1] / height
    scale_z = 6 / depth

    for obj in floors + walls + ceilings:
        obj.select()
    bpy.context.scene.tool_settings.transform_pivot_point = "BOUNDING_BOX_CENTER"
    bpy.ops.transform.resize(
        value=(scale_x, scale_y, scale_z),
        orient_type='GLOBAL',
        orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        orient_matrix_type='GLOBAL',
        constraint_axis=(True, False, False),
        mirror=True,
        use_proportional_edit=False,
        proportional_edit_falloff='SMOOTH',
        proportional_size=1,
        use_proportional_connected=False,
        use_proportional_projected=False,
    )
    for obj in floors + walls + ceilings:
        obj.deselect()
    return


def _load_scenenet(
    file_path: str,
    texture_folder: str,
    label_mapping: LabelIdMapping,
    unknown_texture_folder: str = None,
    timers=None,
) -> List[MeshObject]:
    """ Loads all SceneNet objects at the given "file_path".

    The textures for each object are sampled based on the name of the object, if the name is not represented in the
    texture folder the unknown folder is used. This folder does not exists, after downloading the texture dataset.
    Make sure to create and put some textures, you want to use for these instances there.

    All objects get "category_id" set based on the data in the "resources/id_mappings/nyu_idset.csv"

    Each object will have the custom property "is_scene_net_obj".

    :param file_path: The path to the .obj file from SceneNet.
    :param texture_folder: The path to the texture folder used to sample the textures.
    :param unknown_texture_folder: The path to the textures, which are used if the the texture type is unknown. The default path does not
                                   exist if the dataset was just downloaded, it has to be created manually.
    :return: The list of loaded mesh objects.
    """
    if unknown_texture_folder is None:
        unknown_texture_folder = os.path.join(texture_folder, "unknown")

    # load the objects (Use use_image_search=False as some image names have a "/" prefix which will lead to blender search the whole root directory recursively!
    with timers["load"]:
        if file_path.endswith("gltf") or file_path.endswith("glb"):
            loaded_objects = load_gltf(filepath=file_path)
        else:
            raise ValueError("Unknown file type")

    with timers["sort"]:
        loaded_objects.sort(key=lambda ele: ele.get_name())

    # sample materials for each object
    with timers["materials"]:
        SceneNetLoader._random_sample_materials_for_each_obj(
            loaded_objects,
            texture_folder,
            unknown_texture_folder,
            timers["materials"],
        )

    # set the category ids for each object
    with timers["cat"]:
        SceneNetLoader._set_category_ids(loaded_objects, label_mapping)

    for obj in loaded_objects:
        obj.set_cp("is_scene_net_obj", True)
    return loaded_objects


def load_scenenet(obj_path, texture_path, scales, timers):
    # Load the scenenet room and label its objects with category ids based on the nyu mapping
    label_mapping = bproc.utility.LabelIdMapping.from_csv('nyu_idset.csv')
    with timers["load_scenenet"]:
        objs = _load_scenenet(obj_path, texture_path, label_mapping, timers=timers["load_scenenet"])

    # In some scenes floors, walls and ceilings are one object that needs to be split first
    # Collect all walls
    with timers["filter"]:
        walls    = filter_objs_by_cat(objs, label_mapping, "wall", id_only=True)
        floors   = filter_objs_by_cat(objs, label_mapping, "floor")
        ceilings = filter_objs_by_cat(objs, label_mapping, "ceiling")
        new_objs = walls + floors + ceilings
    others = [o for o in objs if o not in new_objs]

    emission_strength = np.random.uniform(0.5, 1.0)
    bproc.lighting.light_surface(ceilings, emission_strength=emission_strength)

    with timers["rescale"]:
        rescale_scenenet(scales, {
            "walls": walls,
            "floors": floors,
            "ceilings": ceilings,
        })
    with timers["center"]:
        center_scenenet(new_objs)
    return walls, floors, ceilings, others


class SceneNetLoader:

    @staticmethod
    def _random_sample_materials_for_each_obj(loaded_objects: List[MeshObject], texture_folder: str, unknown_texture_folder: str, timers):
        """
        Random sample materials for each of the loaded objects

        Based on the name the textures from the texture_folder will be selected

        :param loaded_objects: objects loaded from the .obj file
        :param texture_folder: The path to the texture folder used to sample the textures.
        :param unknown_texture_folder: The path to the textures, which are used if the the texture type is unknown.
        """
        # for each object add a material
        for obj in loaded_objects:
            for material in obj.get_materials():
                if material is None:
                    continue

                with timers["get_node"]:
                    principled_bsdf = material.get_the_one_node_with_type("BsdfPrincipled")
                    texture_nodes = material.get_nodes_with_type("ShaderNodeTexImage")

                if not texture_nodes or len(texture_nodes) == 1:
                    if len(texture_nodes) == 1:
                        # these materials do not exist they are just named in the .mtl files
                        texture_node = texture_nodes[0]
                    else:
                        texture_node = material.new_node("ShaderNodeTexImage")
                    mat_name = material.get_name()
                    if "." in mat_name:
                        mat_name = mat_name[:mat_name.find(".")]
                    mat_name = mat_name.replace("_", "")
                    # remove all digits from the string
                    mat_name = ''.join([i for i in mat_name if not i.isdigit()])

                    with timers["search"]:
                        image_paths = glob.glob(os.path.join(texture_folder, mat_name, "*"))
                        if not image_paths:
                            if not os.path.exists(unknown_texture_folder):
                                raise Exception("The unknown texture folder does not exist: {}, check if it was "
                                                "set correctly via the config.".format(unknown_texture_folder))

                            image_paths = glob.glob(os.path.join(unknown_texture_folder, "*"))
                            if not image_paths:
                                raise Exception("The unknown texture folder did not contain "
                                                "any textures: {}".format(unknown_texture_folder))
                    image_paths.sort()
                    image_path = random.choice(image_paths)
                    
                    with timers["load"]:
                        if os.path.exists(image_path):
                            texture_node.image = bpy.data.images.load(image_path, check_existing=True)
                        else:
                            raise Exception("No image was found for this entity: {}, "
                                            "material name: {}".format(obj.get_name(), mat_name))

                    with timers["link"]:
                        material.link(texture_node.outputs["Color"], principled_bsdf.inputs["Base Color"])

        # with timers["materials:shading"]:
        #     for obj in loaded_objects:
        #         obj_name = obj.get_name()
        #         if "." in obj_name:
        #             obj_name = obj_name[:obj_name.find(".")]
        #         obj_name = obj_name.lower()
        #         if "wall" in obj_name or "floor" in obj_name or "ceiling" in obj_name:
        #             # set the shading of all polygons to flat
        #             obj.set_shading_mode("FLAT", timers=timers)

    @staticmethod
    def _set_category_ids(loaded_objects: List[MeshObject], label_mapping: LabelIdMapping):
        """
        Set the category ids for the objs based on the .csv file loaded in LabelIdMapping

        Each object will have a custom property with a label, can be used by the SegMapRenderer.

        :param loaded_objects: objects loaded from the .obj file
        """

        #  Some category names in scenenet objects are written differently than in nyu_idset.csv
        normalize_name = {"floor-mat": "floor_mat", "refrigerator": "refridgerator", "shower-curtain": "shower_curtain", 
        "nightstand": "night_stand", "Other-structure": "otherstructure", "Other-furniture": "otherfurniture",
        "Other-prop": "otherprop", "floor_tiles_floor_tiles_0125": "floor", "ground": "floor", "floor_enclose": "floor", "floor_enclose2": "floor",
        "floor_base_object01_56": "floor", "walls1_line01_12": "wall", "room_skeleton": "wall", "ceilingwall": "ceiling"}

        for obj in loaded_objects:
            obj_name = obj.get_name().lower().split(".")[0]

            # If it's one of the cases that the category have different names in both idsets.
            if obj_name in normalize_name:
                obj_name = normalize_name[obj_name]  # Then normalize it.

            if label_mapping.has_label(obj_name):
                obj.set_cp("category_id", label_mapping.id_from_label(obj_name))
            # Check whether the object's name without suffixes like 's', '1' or '2' exist in the mapping.
            elif label_mapping.has_label(obj_name[:-1]):
                obj.set_cp("category_id", label_mapping.id_from_label(obj_name[:-1]))
            elif "painting" in obj_name:
                obj.set_cp("category_id", label_mapping.id_from_label("picture"))
            else:
                print("This object was not specified: {} use objects for it.".format(obj_name))
                obj.set_cp("category_id", label_mapping.id_from_label("otherstructure".lower()))

            # Correct names of floor and ceiling objects to make them later easier to identify (e.g. by the FloorExtractor)
            if obj.get_cp("category_id") == label_mapping.id_from_label("floor"):
                obj.set_name("floor")
            elif obj.get_cp("category_id") == label_mapping.id_from_label("ceiling"):
                obj.set_name("ceiling")

