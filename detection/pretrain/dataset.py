#!/usr/bin/env python
import io
import zarr
import torch
import random
import logging
import numpy as np

import torch.distributed as dist

from pretrain.utils import binary_mask_to_polygon

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.engine import default_setup
from detectron2.config import CfgNode
from detectron2.engine import default_argument_parser, default_setup

from typing import Generator, List, Dict, Union, Optional
from PIL import Image
from zarr import Group
from tqdm import tqdm
from torch import Tensor
from torch.multiprocessing import Process, Queue


def to_numpy(tensor):
    if isinstance(tensor, Tensor):
        return tensor.cpu().numpy()

    if isinstance(tensor, list):
        return [to_numpy(t) for t in tensor]

    if isinstance(tensor, dict):
        return {k: to_numpy(v) for k, v in tensor.items()}
    return tensor


def to_tensor(array):
    if isinstance(array, np.ndarray):
        return torch.as_tensor(array)

    if isinstance(array, list):
        return [to_tensor(a) for a in array]

    if isinstance(array, dict):
        return {k: to_tensor(v) for k, v in array.items()}
    return array


def convert_to_dict(
    bboxes: np.ndarray,
    image: bytes,
    masks: Optional[np.ndarray],
) -> Dict:
    bboxes = bboxes.reshape(-1, 5)
    annotations = [{
        "bbox": bbox[0:4],
        "category_id": bbox[4],
        "bbox_mode": 0
    } for bbox in bboxes]

    image = Image.open(io.BytesIO(image))
    image = utils.convert_PIL_to_numpy(image, "BGR")

    if masks is not None:
        height, width = image.shape[0:2]
        masks = masks.reshape(-1, height, width)

        for i in range(len(annotations)):
            mask = masks[i].astype(np.uint8)
            poly = binary_mask_to_polygon(mask)
            annotations[i]["segmentation"] = poly

    anno = {
        "image": image,
        "annotations": annotations
    }
    return anno


def load_data(zpaths: List[str], data_queue, max_images, cache_size) -> None:
    zfiles = [zarr.open(zpath, "r") for zpath in zpaths]
    if max_images < 0:
        num_images  = [len(z["images"]) for z in zfiles]
        num_chunks  = [z["images"].nchunks for z in zfiles]
        chunk_sizes = [z["images"].chunks[0] for z in zfiles]
    else:
        num_images  = [max_images for z in zfiles]
        chunk_sizes = [z["images"].chunks[0] for z in zfiles]
        num_chunks  = [(n + c - 1) // c for n, c in zip(num_images, chunk_sizes)]

    def shuffle_inds() -> List[List[int]]:
        all_indices = []
        for i, (num_image, num_chunk, chunk_size) in enumerate(
            zip(num_images, num_chunks, chunk_sizes)
        ):
            for chunk in range(num_chunk):
                sid = chunk * chunk_size
                eid = min(sid + chunk_size, num_image)
                all_indices.append((i, sid, eid))
        return np.random.permutation(all_indices).tolist()

    def fetch_data(inds: List[List[int]]) -> Dict[str, List]:
        num_data = 0

        data: Dict[str, List] = {
            "bboxes": [],
            "images": [],
        }

        if "masks" in zfiles[0]:
            data["masks"] = []

        if "depths" in zfiles[0]:
            data["depths"] = []

        while inds and num_data < cache_size:
            i, sid, eid = inds.pop()

            for key in data:
                data[key].append(zfiles[i][key][sid:eid])
            num_data += (eid - sid)
        return data

    all_inds: List[List[int]] = []
    while True:
        if not all_inds:
            all_inds = shuffle_inds()

        data = fetch_data(all_inds)
        for key in data:
            data[key] = [a for d in data[key] for a in d]

        inds = []
        for i in range(2):
            inds += np.random.permutation(len(data["images"])).tolist()

        for i in inds:
            if len(data["bboxes"][i]) == 0 or ("mask" in data and len(data["masks"][i]) == 0):
                continue

            data_queue.put(
                convert_to_dict(
                    data["bboxes"][i],
                    data["images"][i],
                    data["masks"][i] if "masks" in data else None,
                )
            )


class ZarrDataset(object):
    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        train_zarr: str,
        batch_size: int,
        num_queries: int,
        query_size: int,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False,
        random_query: bool = True,
        query_shot: int = 1,
        query_zkey: str = "queries",
        max_images: int = -1,
        cache_size: int = 1250,
    ):
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = augmentations
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.use_keypoint           = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk          = precomputed_proposal_topk
        self.recompute_boxes        = recompute_boxes
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

        print(f"{train_zarr=}")
        self.train_zarr = train_zarr
        self._zpath = train_zarr
        self._bsize = batch_size
        self._num_queries = num_queries
        self._query_size  = query_size
        self._query_shot  = query_shot
        self._query_zkey  = query_zkey
        self._random_query = random_query
        self._max_images   = max_images
        self._cache_size   = cache_size

        self._query_aug = T.AugmentationList([
            T.ResizeShortestEdge(query_size, query_size)
        ])

        zfile = zarr.open(self._zpath, "r")
        assert self._query_shot <= zfile[query_zkey].shape[1], "invalid query shot"

        query_ids = zfile["query_ids"][:]
        bboxes    = zfile["bboxes"][:self._max_images] if self._max_images > -1 else zfile["bboxes"][:]

        cls_ids  = [int(c) for b in bboxes for c in b.reshape(-1, 5)[:, 4]]
        cls_ids += list(map(int, query_ids))
        cls_ids  = sorted(set(cls_ids))

        self.id_map = {cls_id: i for i, cls_id in enumerate(cls_ids)}

    @classmethod
    def from_config(cls, cfg: CfgNode, is_train: bool = True, custom_augs: bool = False) -> Dict:
        augs = utils.build_augmentation(cfg, is_train)

        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        augs = T.AugmentationList(augs)

        ret = {
            "is_train": is_train,
            "batch_size": cfg.SOLVER.IMS_PER_BATCH,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
            "train_zarr": cfg.DATASETS.TRAIN_ZARR,
            "num_queries": cfg.MODEL.NUM_QUERIES,
            "query_size": cfg.MODEL.QUERY_SIZE,
            "query_shot": cfg.MODEL.QUERY_SHOT,
            "random_query": cfg.MODEL.RANDOM_QUERY,
            "query_zkey": cfg.MODEL.QUERY_ZKEY,
            "max_images": cfg.DATASETS.MAX_IMAGES,
        }

        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)

        if cfg.MODEL.LOAD_PROPOSALS:
            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        return ret

    def __len__(self) -> int:
        zfile = zarr.open(self._zpath, "r")
        return len(zfile["images"])

    def _test(self) -> Generator[List[Dict], None, None]:
        zfile = zarr.open(self._zpath, "r")

        images = zfile["images"]
        bboxes = zfile["bboxes"]
        queries    = zfile[self._query_zkey][:]
        query_ids  = zfile["query_ids"]
        chunk_size = zfile["images"].chunks[0]

        if not hasattr(self, "aug_queries"):
            aug_queries  = []
            query_bboxes = []
            for i in range(queries.shape[0]):
                for j in range(queries.shape[1]):
                    query = Image.open(io.BytesIO(queries[i, j]))
                    query = utils.convert_PIL_to_numpy(query, "BGR")

                    aug_query = T.AugInput(query)
                    self._query_aug(aug_query)
                    aug_query = torch.as_tensor(
                        np.ascontiguousarray(aug_query.image.transpose(2, 0, 1))
                    )

                    query_bboxes.append([0, 0, aug_query.shape[2], aug_query.shape[1]])  # x1, y1, x2, y2
                    aug_queries.append(aug_query)
            self.aug_queries = aug_queries
            self.query_bboxes = query_bboxes

        chunks = (len(images) + chunk_size - 1) // chunk_size
        for i in range(chunks):
            sind = i * chunk_size
            eind = min(sind + chunk_size, len(images))

            _images = images[sind:eind]
            _bboxes = bboxes[sind:eind]
            for j, (image, bbox) in enumerate(zip(_images, _bboxes)):
                mask  = None  # if masks is None else masks[i]
                dataset_dict = convert_to_dict(bbox.reshape(-1, 5), image, mask)
                dataset_dict["image_id"]  = j + sind
                dataset_dict["queries"]   = self.aug_queries
                dataset_dict["query_ids"] = query_ids
                dataset_dict["query_bboxes"] = self.query_bboxes
                self._prepare_data(dataset_dict)
                dataset_dict["query_ids"] = torch.as_tensor(dataset_dict["query_ids"])
                dataset_dict["query_bboxes"] = torch.as_tensor(dataset_dict["query_bboxes"])

                yield [dataset_dict]

    def _prepare_data(self, dataset_dict: Dict) -> Dict[str, List]:
        image = dataset_dict["image"]
        utils.check_image_size(dataset_dict, image)

        aug_input  = T.AugInput(image, sem_seg=None)
        transforms = self.augmentations(aug_input)
        image = aug_input.image

        image_shape = image.shape[:2]
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(
                image.transpose(2, 0, 1)
            )
        )

        for anno in dataset_dict["annotations"]:
            anno["category_id"] = self.id_map[anno["category_id"]]
        dataset_dict["query_ids"] = [self.id_map[q] for q in dataset_dict["query_ids"]]

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices
                ) for obj in dataset_dict.pop("annotations")
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(
                instances, box_threshold=8
            )
        return dataset_dict

    def _load_queries(self, zfile: Group) -> Dict[int, List[np.ndarray]]:
        results: Dict[int, List[np.ndarray]] = {}

        queries   = zfile[self._query_zkey][:]
        query_ids = zfile["query_ids"][:]

        for queries, query_id in tqdm(zip(queries, query_ids), total=len(queries)):
            results[query_id] = []
            for query in queries:
                if query is None:
                    continue
                results[query_id].append(query)
        return results

    def _load_images(self, data_queue, queries, world_size):
        pos_query_ids = []

        images = []
        for _ in range(self._bsize // world_size):
            dataset_dict = data_queue.get(block=True)
            for anno in dataset_dict["annotations"]:
                pos_query_ids.append(anno["category_id"])
            images.append(dataset_dict)

        pos_query_ids = set(pos_query_ids)
        neg_query_ids = set(queries.keys()) - pos_query_ids
        pos_query_ids = list(pos_query_ids)
        neg_query_ids = list(neg_query_ids)

        random.shuffle(pos_query_ids)
        random.shuffle(neg_query_ids)

        aug_queries  = []
        query_ids    = []
        query_bboxes = []

        _query_ids = pos_query_ids + neg_query_ids
        _query_ids = _query_ids[::-1]
        while len(aug_queries) < self._num_queries * self._query_shot and _query_ids:
            query_id = _query_ids.pop()

            if len(queries[query_id]) == 0:
                continue

            rinds = np.random.permutation(len(queries[query_id]))[:self._query_shot]
            for rind in rinds:
                aug_query = queries[query_id][rind]

                aug_query = utils.convert_PIL_to_numpy(Image.open(io.BytesIO(aug_query)), "BGR")
                aug_query = T.AugInput(aug_query)
                self._query_aug(aug_query)
                aug_query = torch.as_tensor(
                    np.ascontiguousarray(
                        aug_query.image.transpose(2, 0, 1)
                    )
                )

                query_ids.append(query_id)
                query_bboxes.append([
                    0, 0, aug_query.shape[2], aug_query.shape[1]
                ])  # x1, y1, x2, y2
                aug_queries.append(aug_query)

        images[0]["queries"] = aug_queries
        for i in range(len(images)):
            images[i]["query_ids"]    = query_ids
            images[i]["query_bboxes"] = query_bboxes
            self._prepare_data(images[i])
            images[i]["query_ids"]    = torch.as_tensor(images[i]["query_ids"])
            images[i]["query_bboxes"] = torch.as_tensor(images[i]["query_bboxes"])

        return images

    def start_fetching(self, data_queue, output_queue, world_size):
        zfile   = zarr.open(self._zpath, "r")
        queries = self._load_queries(zfile)
        while True:
            images = self._load_images(data_queue, queries, world_size)
            images = to_numpy(images)
            output_queue.put(images)


def fetch_data(
    data_queue,
    output_queue,
    world_size,
    is_train: bool,
    train_zarr: str,
    batch_size: int,
    num_queries: int,
    query_size: int,
    augmentations: List[Union[T.Augmentation, T.Transform]],
    image_format: str,
    use_instance_mask: bool = False,
    use_keypoint: bool = False,
    instance_mask_format: str = "polygon",
    keypoint_hflip_indices: Optional[np.ndarray] = None,
    precomputed_proposal_topk: Optional[int] = None,
    recompute_boxes: bool = False,
    random_query: bool = True,
    query_shot: int = 1,
    query_zkey: str = "queries",
    max_images: int = -1,
    cache_size: int = 1250,
):
    dataset = ZarrDataset(
        is_train=is_train,
        train_zarr=train_zarr,
        batch_size=batch_size,
        num_queries=num_queries,
        query_size=query_size,
        augmentations=augmentations,
        image_format=image_format,
        use_instance_mask=use_instance_mask,
        use_keypoint=use_keypoint,
        instance_mask_format=instance_mask_format,
        keypoint_hflip_indices=keypoint_hflip_indices,
        precomputed_proposal_topk=precomputed_proposal_topk,
        recompute_boxes=recompute_boxes,
        random_query=random_query,
        query_shot=query_shot,
        query_zkey=query_zkey,
        max_images=max_images,
        cache_size=cache_size,
    )
    dataset.start_fetching(data_queue, output_queue, world_size)


class ZarrWrapper(ZarrDataset):
    def __iter__(self):
        # with Manager() as manager:
        data_queue   = Queue(64)
        data_process = Process(
            target=load_data,
            args=(
                [self._zpath],
                data_queue,
                self._max_images,
                self._cache_size,
            ),
        )
        data_process.start()

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        output_queue  = Queue(8)
        fetch_process = Process(
            target=fetch_data,
            args=(
                data_queue,
                output_queue,
                world_size,
                self.is_train,
                self.train_zarr,
                self._bsize,
                self._num_queries,
                self._query_size,
                self.augmentations,
                self.image_format,
                self.use_instance_mask,
                self.use_keypoint,
                self.instance_mask_format,
                self.keypoint_hflip_indices,
                self.proposal_topk,
                self.recompute_boxes,
                self._random_query,
                self._query_shot,
                self._query_zkey,
                self._max_images,
                self._cache_size,
            ),
        )
        fetch_process.start()

        try:
            while True:
                images = output_queue.get(block=True)
                images = to_tensor(images)
                yield images
        finally:
            fetch_process.terminate()
            data_process.terminate()
