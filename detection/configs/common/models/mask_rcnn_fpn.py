from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detectron2.modeling.backbone import BasicStem, FPN, ResNet
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator import RPN, StandardRPNHead
from detectron2.modeling.roi_heads import (
    StandardROIHeads,
    FastRCNNOutputLayers,
    MaskRCNNConvUpsampleHead,
    FastRCNNConvFCHead,
)

from pretrain.model import Wrapper
from pretrain.roi_heads import (
    FastRCNNOutputLayersTarget,
    FastRCNNOutputLayersQuery,
    ROIHeadsTarget,
    ROIHeadsQuery,
)

model = L(Wrapper)(
    backbone_q=L(FPN)(
        bottom_up=L(ResNet)(
            stem=L(BasicStem)(
                in_channels=3,
                out_channels=64,
                norm="SyncBN",
            ),
            stages=L(ResNet.make_default_stages)(
                depth=50,
                stride_in_1x1=True,
                norm="SyncBN",
            ),
            out_features=[
                "res2",
                "res3",
                "res4",
                "res5",
            ],
        ),
        in_features="${.bottom_up.out_features}",
        out_channels=256,
        top_block=L(LastLevelMaxPool)(),
    ),
    backbone_k=L(FPN)(
        bottom_up=L(ResNet)(
            stem=L(BasicStem)(
                in_channels=3,
                out_channels=64,
                norm="SyncBN",
            ),
            stages=L(ResNet.make_default_stages)(
                depth=50,
                stride_in_1x1=True,
                norm="SyncBN",
            ),
            out_features=[
                "res2",
                "res3",
                "res4",
                "res5",
            ],
        ),
        in_features="${.bottom_up.out_features}",
        out_channels=256,
        top_block=L(LastLevelMaxPool)(),
    ),
    proposal_generator_q=L(RPN)(
        in_features=["p2", "p3", "p4", "p5", "p6"],
        head=L(StandardRPNHead)(in_channels=256, num_anchors=3),
        anchor_generator=L(DefaultAnchorGenerator)(
            sizes=[[32], [64], [128], [256], [512]],
            aspect_ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
            offset=0.0,
        ),
        anchor_matcher=L(Matcher)(
            thresholds=[0.3, 0.7], labels=[0, -1, 1], allow_low_quality_matches=True
        ),
        box2box_transform=L(Box2BoxTransform)(weights=[1.0, 1.0, 1.0, 1.0]),
        batch_size_per_image=256,
        positive_fraction=0.5,
        pre_nms_topk=(2000, 1000),
        post_nms_topk=(1000, 1000),
        nms_thresh=0.7,
    ),
    proposal_generator_k=L(RPN)(
        in_features=["p2", "p3", "p4", "p5", "p6"],
        head=L(StandardRPNHead)(in_channels=256, num_anchors=3),
        anchor_generator=L(DefaultAnchorGenerator)(
            sizes=[[32], [64], [128], [256], [512]],
            aspect_ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
            offset=0.0,
        ),
        anchor_matcher=L(Matcher)(
            thresholds=[0.3, 0.7], labels=[0, -1, 1], allow_low_quality_matches=True
        ),
        box2box_transform=L(Box2BoxTransform)(weights=[1.0, 1.0, 1.0, 1.0]),
        batch_size_per_image=256,
        positive_fraction=0.5,
        pre_nms_topk=(2000, 1000),
        post_nms_topk=(1000, 1000),
        nms_thresh=0.7,
    ),
    roi_heads_q=L(ROIHeadsTarget)(
        num_classes="${model.num_classes}",
        batch_size_per_image=512,
        positive_fraction=0.25,
        proposal_matcher=L(Matcher)(
            thresholds=[0.5], labels=[0, 1], allow_low_quality_matches=False
        ),
        box_in_features=["p2", "p3", "p4", "p5"],
        box_pooler=L(ROIPooler)(
            output_size=7,
            scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        ),
        box_head=L(FastRCNNConvFCHead)(
            input_shape=ShapeSpec(channels=256, height=7, width=7),
            conv_dims=[],
            fc_dims=[1024, 1024],
        ),
        box_predictor=L(FastRCNNOutputLayersTarget)(
            input_shape=ShapeSpec(channels=1024),
            test_score_thresh=0.05,
            box2box_transform=L(Box2BoxTransform)(weights=(10, 10, 5, 5)),
            num_classes="${..num_classes}",
            temperature=0.2,
            query_shot=1,
            cls_agnostic_bbox_reg=True,
        ),
        mask_in_features=["p2", "p3", "p4", "p5"],
        mask_pooler=L(ROIPooler)(
            output_size=14,
            scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        ),
        mask_head=L(MaskRCNNConvUpsampleHead)(
            input_shape=ShapeSpec(channels=256, width=14, height=14),
            num_classes=1,
            conv_dims=[256, 256, 256, 256, 256],
        ),
    ),
    roi_heads_k=L(ROIHeadsQuery)(
        num_classes="${model.num_classes}",
        batch_size_per_image=512,
        positive_fraction=0.25,
        proposal_matcher=L(Matcher)(
            thresholds=[0.5],
            labels=[0, 1],
            allow_low_quality_matches=False,
        ),
        box_in_features=["p2", "p3", "p4", "p5"],
        box_pooler=L(ROIPooler)(
            output_size=7,
            scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        ),
        box_head=L(FastRCNNConvFCHead)(
            input_shape=ShapeSpec(channels=256, height=7, width=7),
            conv_dims=[],
            fc_dims=[1024, 1024],
        ),
        box_predictor=L(FastRCNNOutputLayersQuery)(
            input_shape=ShapeSpec(channels=1024),
            test_score_thresh=0.05,
            box2box_transform=L(Box2BoxTransform)(weights=(10, 10, 5, 5)),
            num_classes="${..num_classes}",
            temperature="${model.roi_heads_q.box_predictor.temperature}",
            query_shot="${model.roi_heads_q.box_predictor.query_shot}",
            cls_agnostic_bbox_reg="${model.roi_heads_q.box_predictor.cls_agnostic_bbox_reg}",
        ),
    ),
    num_classes=52447,
    momentum=0.999,
    pixel_mean=[103.530, 116.280, 123.675],
    pixel_std=[1.0, 1.0, 1.0],
    input_format="BGR",
    eman=True,
)
