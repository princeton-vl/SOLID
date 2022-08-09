from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L

from pretrain.dataset import ZarrWrapper

dataloader = OmegaConf.create()
dataloader.train = L(ZarrWrapper)(
    is_train=True,
    train_zarr="./datasets/SOLID.zarr",
    batch_size=128,
    num_queries=128,
    query_size=224,
    augmentations=L(T.AugmentationList)(
        augs=[
            L(T.ResizeShortestEdge)(
                short_edge_length=(640, 672, 704, 736, 768, 800),
                sample_style="choice",
                max_size=1333,
            ),
            L(T.RandomFlip)(horizontal=True),
        ],
    ),
    image_format="BGR",
    random_query=True,
    query_shot=1,
    query_zkey="queries",
    recompute_boxes=True,
    use_instance_mask=True,
    max_images=1000000,
)
