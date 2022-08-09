import detectron2.data.transforms as T
from detectron2.config.lazy import LazyCall as L
from detectron2.layers.batch_norm import NaiveSyncBatchNorm
from detectron2.solver import WarmupParamScheduler
from fvcore.common.param_scheduler import CosineParamScheduler

from ..common.data.pretrain import dataloader
from ..common.models.mask_rcnn_fpn import model
from ..common.optim import SGD as optimizer
from ..common.train import train

# train from scratch
train.init_checkpoint = ""
train.amp.enabled = True
train.ddp.fp16_compression = True
model.backbone_q.bottom_up.freeze_at = 0
model.backbone_k.bottom_up.freeze_at = 0

model.backbone_k.bottom_up.stem.norm = \
    model.backbone_k.bottom_up.stages.norm = \
    model.backbone_k.norm = "BN"
model.backbone_q.bottom_up.stem.norm = \
    model.backbone_q.bottom_up.stages.norm = \
    model.backbone_q.norm = "BN"

model.proposal_generator_q.head.conv_dims = [-1]
model.proposal_generator_k.head.conv_dims = [-1]

model.pixel_mean = [71.42, 84.02, 93.79]
model.pixel_std  = [48.09, 47.98, 47.64]

train.max_iter = 1000000
lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(CosineParamScheduler)(
        start_value=1.0,
        end_value=0.01,
    ),
    warmup_length=min(10000 / train.max_iter, 1.0),
    warmup_factor=0.0001,
)

optimizer.lr = 0.1
optimizer.weight_decay = 4e-5

train.eval_period = 0
