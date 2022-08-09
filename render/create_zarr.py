#!/usr/bin/env python
import zarr
import argparse

CHUNK_SIZE = 256


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("zarr_path", type=str)
    parser.add_argument("--num_classes", default=100, type=int)
    parser.add_argument("--num_images", default=int(1e6), type=int)
    parser.add_argument("--num_shots", default=1, type=int)
    parser.add_argument("--num_res", default=1, type=int)
    return parser.parse_args()


def init_stats(zfile, num_images, num_res):
    keys = [
        "visibilities",
        "mask_bbox_ratios",
        "angles",
    ]

    for key in keys:
        for i in range(num_res):
            zfile.empty(
                key if num_res == 1 else f"{i}/{key}",
                shape=num_images,
                dtype="array:f4",
                chunks=CHUNK_SIZE,
            )


def main():
    args = parse_args()

    zarr_path   = args.zarr_path
    num_shots   = args.num_shots
    num_images  = args.num_images
    num_classes = args.num_classes
    num_res     = args.num_res

    zfile = zarr.open(zarr_path, "a")

    for i in range(num_res):
        key_valids  = f"{i}/valids" if num_res > 1 else "valids"
        key_images  = f"{i}/images" if num_res > 1 else "images"
        key_bboxes  = f"{i}/bboxes" if num_res > 1 else "bboxes"
        key_masks   = f"{i}/masks"  if num_res > 1 else "masks"
        key_queries = f"{i}/queries" if num_res > 1 else "queries"
        key_qids    = f"{i}/query_ids" if num_res > 1 else "query_ids"
        key_models  = f"{i}/scenenet_models" if num_res > 1 else "scenenet_models"

        zfile.zeros(
            key_valids,
            shape=num_images,
            dtype=bool,
        )

        zfile.empty(
            key_images,
            shape=num_images,
            dtype="array:i1",
            chunks=CHUNK_SIZE,
            compressor=None,
        )

        zfile.empty(
            key_bboxes,
            shape=num_images,
            dtype="array:i4",
            chunks=CHUNK_SIZE,
        )

        zfile.empty(
            key_masks,
            shape=num_images,
            dtype="array:i1",
            chunks=CHUNK_SIZE,
        )

        zfile.empty(
            key_queries,
            shape=(num_classes, num_shots),
            dtype="array:i1",
            chunks=CHUNK_SIZE,
            compressor=None,
        )

        zfile.zeros(
            key_qids,
            shape=num_classes,
            dtype=int,
        )
        zfile[key_qids] = list(range(num_classes))

        zfile.empty(
            key_models,
            shape=num_images,
            dtype="S128",
        )

    init_stats(zfile, num_images, num_res)


if __name__ == "__main__":
    main()
