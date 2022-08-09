import blenderproc as bproc
import bpy
import sys
import os
import zarr
import numpy as np
import argparse
import traceback

sys.path.insert(0, os.getcwd())
sys.setrecursionlimit(10000)

from rendering.solid import render
from rendering.query import render
from rendering.utils.data import read_objects, buffer_encode_png, masks_to_bboxes
from rendering.utils.argparse import load_parsers, get_render_func

from PIL import Image
from tqdm import tqdm
from blenderproc.python.writer.CocoWriterUtility import CocoWriterUtility


bpy.context.scene.render.use_persistent_data = True
bpy.context.scene.cycles.debug_use_spatial_splits = True

bpy.context.scene.render.tile_x = 256
bpy.context.scene.render.tile_y = 256


def parse_args():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(
        help='sub-command help',
        dest="configuration",
    )
    load_parsers(subparsers)

    parser.add_argument("obj_json", type=str)
    parser.add_argument("zarr_path", type=str)
    parser.add_argument(
        "--target",
        default="targets",
        choices=[
            "targets",
            "queries"
        ],
        type=str
    )
    parser.add_argument("--index", default=0, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument(
        "--model_dir",
        default="/n/fs/pvl-progen/assets_v2/Thingi10K/raw_meshes",
        type=str,
    )
    parser.add_argument(
        "--texture_dir",
        default="/n/fs/pvl-progen/assets_v2/texturehaven",
        type=str,
    )
    parser.add_argument(
        "--scenenet_dir",
        default="/n/fs/pvl-progen/assets_v2/scenenet_cleanup",
        type=str,
    )
    parser.add_argument(
        "--scenenet_texture_dir",
        default="/n/fs/pvl-progen/assets_v2/texture_library",
        type=str,
    )
    parser.add_argument("--save_blend", action="store_true")

    args   = parser.parse_args()
    render = get_render_func(args.configuration)
    return args, render


def main():
    args, render = parse_args()
    target = args.target

    if target == "targets":
        render_scenes(args, render)
    elif target == "queries":
        render_queries(args, render)
    else:
        raise ValueError("unknown target")


def render_scenes(args, render):
    scales = args.scales
    num_scales = len(scales)

    key_images = "0/images" if num_scales > 1 else "images"

    # Zarr settings
    zfile = zarr.open(args.zarr_path, "a")
    num_samples = len(zfile[key_images])
    chunk_size  = zfile[key_images].chunks[0]
    sind = args.index * chunk_size
    eind = min(sind + chunk_size, num_samples)
    save_blend = args.save_blend

    if sind > eind:
        return 0

    key_valids = "0/valids" if num_scales > 1 else "valids"
    valids = zfile[key_valids][:]
    for i in range(1, num_scales):
        key_valids = f"{i}/valids"
        valids &= zfile[key_valids][:]

    idxs_to_write = np.arange(sind, eind)[~valids[sind:eind]]
    batch_size  = args.batch_size
    num_batches = (len(idxs_to_write) + batch_size - 1) // batch_size

    # Rendering settings
    objects = read_objects(args.obj_json)

    for i in tqdm(range(num_batches)):
        _sind = i * batch_size
        _eind = min(_sind + batch_size, len(idxs_to_write))

        all_images = [[] for _ in range(num_scales)]
        all_bboxes = [[] for _ in range(num_scales)]
        all_masks  = [[] for _ in range(num_scales)]
        all_stats  = [{} for _ in range(num_scales)]

        idxs = idxs_to_write[_sind:_eind]
        for j in idxs:
            while True:
                try:
                    data = render(
                        objects,
                        **vars(args),
                    )
                    break
                except Exception as e:
                    traceback.print_exc()
                    continue

            if save_blend:
                filepath = os.path.join("blends", f"{j:03d}.blend")
                bpy.ops.wm.save_as_mainfile(filepath=filepath)

            for i in range(num_scales):
                images = data[i][0]["images"]
                bboxes = data[i][0]["bboxes"]
                masks  = data[i][0]["masks"]

                image = Image.fromarray(images[0])

                all_images[i].append(buffer_encode_png(image))
                all_bboxes[i].append(bboxes.reshape(-1))
                all_masks[i].append(masks.reshape(-1))

                for stat_key in data[i][1]:
                    all_stats[i][stat_key] = all_stats[i].get(
                        stat_key,
                        [],
                    ) + [data[i][1][stat_key]]

        sind = idxs[0]
        eind = idxs[0] + len(idxs)
        for i in range(num_scales):
            key_valids = f"{i}/valids" if num_scales > 1 else "valids"
            key_images = f"{i}/images" if num_scales > 1 else "images"
            key_bboxes = f"{i}/bboxes" if num_scales > 1 else "bboxes"
            key_masks  = f"{i}/masks"  if num_scales > 1 else "masks"

            zfile[key_images][sind:eind] = np.array(
                all_images[i] + [None],
                dtype=object,
            )[:-1]
            zfile[key_bboxes][sind:eind] = np.array(
                all_bboxes[i] + [None],
                dtype=object,
            )[:-1]

            if key_masks in zfile:
                zfile[key_masks][sind:eind] = np.array(
                    all_masks[i] + [None],
                    dtype=object,
                )[:-1]

            for stat_key in all_stats[i]:
                save_key = f"{i}/{stat_key}" if num_scales > 1 else stat_key
                if save_key not in zfile:
                    continue

                zfile[save_key][sind:eind] = np.array(
                    all_stats[i][stat_key] + [None],
                    dtype=object,
                )[:-1]

            zfile[key_valids][sind:eind] = True


def render_queries(args, render):
    zfile   = zarr.open(args.zarr_path, "a")
    objects = read_objects(args.obj_json)

    index = args.index
    chunk_size = zfile["queries"].chunks[0]

    all_images = []
    sind = index * chunk_size
    eind = min(sind + chunk_size, len(objects))
    for i in range(sind, eind):
        try:
            colors, instance_segmaps, instance_attribute_maps = render(
                [objects[i]], **vars(args)
            )

            images = []
            for j in range(len(colors)):
                image = Image.fromarray(colors[j])
                for instance_attribute_map in instance_attribute_maps[j]:
                    if instance_attribute_map["category_id"] == 0:
                        continue

                    mask = (instance_segmaps[j] == instance_attribute_map["idx"])
                    bbox = CocoWriterUtility.bbox_from_binary_mask(mask)
                    bbox[2] += bbox[0]
                    bbox[3] += bbox[1]

                image = image.crop(bbox)
                images.append(buffer_encode_png(image))
        except Exception as e:
            traceback.print_exc()
            images = [None] * zfile.queries.shape[1]
        all_images.append(np.array(images + [None], dtype=object))

    if all_images:
        all_images = np.array(all_images, dtype=object)[:, :-1]
        zfile["queries"][sind:eind] = all_images


if __name__ == "__main__":
    main()
