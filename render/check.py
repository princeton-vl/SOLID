#!/usr/bin/env python
import io
import cv2
import zarr
import argparse
import numpy as np
import os

from PIL import Image, ImageDraw
from tqdm import tqdm
from typing import List

from multiprocessing import Manager, Pool


def parse_args():
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(
        help="sub-command help",
        dest="action",
    )

    parser_verify_dataset = sub_parsers.add_parser("verify_dataset")
    parser_verify_dataset.add_argument("inp_path", type=str)
    parser_verify_dataset.add_argument("oup_path", type=str)
    parser_verify_dataset.add_argument("--sind", default=0, type=int)
    parser_verify_dataset.add_argument("--eind", default=None, type=int)

    parser_visualize_dataset = sub_parsers.add_parser("visualize_dataset")
    parser_visualize_dataset.add_argument("inp_path", type=str)
    parser_visualize_dataset.add_argument("oup_path", type=str)
    parser_visualize_dataset.add_argument("--nums", default=10, type=int)
    parser_visualize_dataset.add_argument("--offs", default=0, type=int)
    parser_visualize_dataset.add_argument("--no_bboxes", action="store_true")

    parser_verify_queries = sub_parsers.add_parser("verify_queries")
    parser_verify_queries.add_argument("inp_path", type=str)

    parser_visualize_queries = sub_parsers.add_parser("visualize_queries")
    parser_visualize_queries.add_argument("inp_path", type=str)
    parser_visualize_queries.add_argument("oup_path", type=str)
    parser_visualize_queries.add_argument("--nums", default=16, type=int)
    return parser.parse_args()


def check_chunk(
    zpath: str,
    sind: int,
    eind: int,
    missings: List[int],
):
    zfile = zarr.open(zpath, "r")

    images = zfile.images[sind:eind]
    bboxes = zfile.bboxes[sind:eind]
    masks  = zfile.masks[sind:eind]

    for j, (image, bbox, mask) in enumerate(zip(images, bboxes, masks)):
        if image.shape[0] == 0:
            missings.append(sind + j)
            continue

        image = Image.open(io.BytesIO(image))
        bbox  = bbox.reshape(-1, 5)
        mask  = mask.reshape(-1, image.height, image.width)

        if bbox.shape[0] != mask.shape[0]:
            missings.append(sind + j)
            continue


def verify_dataset(
    zpath: str,
    opath: str,
    sind: int = 0,
    eind: int = None,
):
    zfile = zarr.open(zpath, "r")

    num_chunks = zfile.images.nchunks
    chunk_size = zfile.images.chunks[0]
    num_images = zfile.images.size

    sind = sind
    eind = eind if eind is not None else num_images

    num_chunks = ((eind - sind) + chunk_size - 1) // chunk_size

    args = []
    with Manager() as manager:
        missings = manager.list()
        for i in range(num_chunks):
            sind_ = sind + i * chunk_size
            eind_ = min(sind_ + chunk_size, eind)
            args.append((zpath, sind_, eind_, missings))

        pbar = tqdm(total=len(args))

        def update(*a):
            pbar.update()

        with Pool(processes=4) as pool:
            for arg in args:
                pool.apply_async(check_chunk, arg, callback=update)
            pool.close()
            pool.join()
            # pool.starmap(check_chunk, args)

        missings = list(map(str, sorted(missings)))
        with open(opath, "w") as f:
            f.write("\n".join(missings))

    zfile    = zarr.open(zpath, "a")
    missings = list(map(int, missings))
    for missing in missings:
        zfile["valids"][missing] = False


def visualize_dataset(
    path: str,
    oupp: str,
    nums: int = 10,
    offs: int = 0,
    no_bboxes: bool = False,
):
    zfile = zarr.open(path, "r")
    images = zfile["images"][offs:offs + nums]
    bboxes = zfile["bboxes"][offs:offs + nums]
    masks  = zfile["masks"][offs:offs + nums]
    if not no_bboxes:
        ratios = zfile["mask_bbox_ratios"][offs:offs + nums]
        visibilities = zfile["visibilities"][offs:offs + nums]

    for i in tqdm(range(nums)):
        image = Image.open(io.BytesIO(images[i]))
        bbox  = bboxes[i].reshape(-1, 5)

        height = image.height
        width  = image.width

        mask = masks[i].reshape(-1, height, width)

        if not no_bboxes:
            ratio = ratios[i]
            visibility = visibilities[i]

            draw = ImageDraw.Draw(image)
            for j, (b, r, v) in enumerate(zip(bbox, ratio, visibility)):
                b = list(map(int, b))

                text = f"{b[4]}/{r:0.2f}/{v:0.2f}"

                draw.rectangle(b[0:4])
                draw.text(b[0:2], text)
        image.save(os.path.join(oupp, f"image_{i:03d}.png"), "png")

        for j, mask in enumerate(mask):
            mask = (mask * 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(
                    oupp,
                    f"mask_{i:03d}_{j:03d}.png",
                ),
                mask
            )


def verify_queries(
    inp_path: str,
):
    zfile = zarr.open(inp_path, "r")

    chunk_size  = zfile.queries.chunks[0]
    num_queries = zfile.queries.shape[0]
    num_chunks  = (num_queries + chunk_size - 1) // chunk_size

    missings = []
    for i in tqdm(range(num_chunks)):
        sind = i * chunk_size
        eind = min(sind + chunk_size, num_queries)
        for j, query in enumerate(zfile.queries[sind:eind]):
            try:
                Image.open(io.BytesIO(query[0]))
            except Exception:
                missings.append(sind + j)
                tqdm.write(str(sind + j))

    missings = list(map(str, missings))
    with open('missings.txt', 'w') as f:
        f.write("\n".join(missings))


def visualize_queries(
    inp_path: str,
    oup_path: str,
    nums: int,
):
    zfile   = zarr.open(inp_path, "r")
    queries = zfile.queries[:]
    for i in tqdm(range(nums)):
        n = queries[i].size
        for j in range(n):
            query = queries[i, j]
            query = Image.open(io.BytesIO(query))
            query.save(
                os.path.join(
                    oup_path,
                    f"{i:03d}_{j:02d}.png",
                ),
                "png",
            )


def main():
    args = parse_args()

    if args.action == "verify_dataset":
        verify_dataset(
            args.inp_path,
            args.oup_path,
            args.sind,
            args.eind,
        )
    elif args.action == "visualize_dataset":
        visualize_dataset(
            args.inp_path,
            args.oup_path,
            args.nums,
            args.offs,
            args.no_bboxes,
        )
    elif args.action == "verify_queries":
        verify_queries(
            args.inp_path,
        )
    elif args.action == "visualize_queries":
        visualize_queries(
            args.inp_path,
            args.oup_path,
            args.nums,
        )
    else:
        raise ValueError("unknown action")


if __name__ == "__main__":
    main()
