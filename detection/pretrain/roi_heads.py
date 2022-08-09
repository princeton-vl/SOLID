# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# This file contains code under Apache-2.0 License from https://github.com/facebookresearch/detectron2
import torch
import torch.nn as nn

from typing import Dict, List, Optional, Tuple, Union

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, cat, cross_entropy
from detectron2.modeling import StandardROIHeads, ROI_HEADS_REGISTRY
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import ImageList, Instances, Boxes
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers


@ROI_HEADS_REGISTRY.register()
class ROIHeadsTarget(StandardROIHeads):
    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        outputs = StandardROIHeads._init_box_head(cfg, input_shape)
        outputs["box_predictor"] = FastRCNNOutputLayersTarget(cfg, outputs["box_head"].output_shape)
        return outputs

    def _forward_box(
        self, features: Dict[str, torch.Tensor], proposals: List[Instances],
    ):
        queries   = features["queries"]
        query_ids = features["query_ids"]

        box_features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(box_features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)

        _keep_inds = None
        if self.training:
            _gt_classes = cat([
                p.gt_classes for p in proposals
            ], dim=0) if len(proposals) else torch.empty(0)
            _query_ids  = query_ids.reshape(-1, self.box_predictor.query_shot)[:, 0]
            _gt_classes = (_gt_classes.reshape(-1, 1) == _query_ids.reshape(1, -1))
            _keep_inds  = _gt_classes.any(dim=1)
        predictions = self.box_predictor(box_features, queries, _keep_inds)
        del box_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals, query_ids, _keep_inds)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            losses["predictions"] = predictions
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances


class FastRCNNOutputLayersTarget(FastRCNNOutputLayers):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        box2box_transform,
        num_classes: int,
        test_score_thresh: float = 0.0,
        test_nms_thresh: float = 0.5,
        test_topk_per_image: int = 100,
        cls_agnostic_bbox_reg: bool = False,
        smooth_l1_beta: float = 0.0,
        box_reg_loss_type: str = "smooth_l1",
        loss_weight: Union[float, Dict[str, float]] = 1.0,
        temperature: float = 1,
        embd_dims: List[int] = [256],
        query_shot: int = 1,
    ):
        super(FastRCNNOutputLayersTarget, self).__init__(
            input_shape,
            box2box_transform=box2box_transform,
            num_classes=num_classes,
            test_score_thresh=test_score_thresh,
            test_nms_thresh=test_nms_thresh,
            test_topk_per_image=test_topk_per_image,
            cls_agnostic_bbox_reg=cls_agnostic_bbox_reg,
            smooth_l1_beta=smooth_l1_beta,
            box_reg_loss_type=box_reg_loss_type,
            loss_weight=loss_weight,
        )
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        self.temperature = temperature
        self.query_shot  = query_shot

        cls_score = [nn.Linear(input_size, embd_dims[0])]
        for prev_dim, curr_dim in zip(embd_dims[:-1], embd_dims[1:]):
            cls_score += [
                nn.ReLU(inplace=True),
                nn.Linear(prev_dim, curr_dim),
            ]
        self.cls_score = nn.Sequential(*cls_score) if len(cls_score) > 1 else cls_score[0]

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            # fmt: off
            "num_classes"           : cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "cls_agnostic_bbox_reg" : cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "smooth_l1_beta"        : cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh"     : cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh"       : cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image"   : cfg.TEST.DETECTIONS_PER_IMAGE,
            "box_reg_loss_type"     : cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
            "loss_weight"           : {
                "loss_box_reg": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT,
                "loss_cls": cfg.MODEL.ROI_BOX_HEAD.CLS_LOSS_WEIGHT
            },
            "temperature"           : cfg.MODEL.ROI_HEADS.TEMPERATURE,
            "embd_dims"             : cfg.MODEL.ROI_HEADS.EMBD_DIMS,
            "query_shot"            : cfg.MODEL.QUERY_SHOT,
            # fmt: on
        }

    def forward(self, features, queries, keep_inds=None):
        if features.dim() > 2:
            features = torch.flatten(features, start_dim=1)

        scores  = scores if keep_inds is None else features[keep_inds]

        queries = nn.functional.normalize(queries, dim=1)
        scores  = nn.functional.normalize(self.cls_score(scores), dim=1)
        scores  = torch.einsum("ij,kj->ik", scores, queries) / self.temperature

        scores = scores.reshape(scores.shape[0], -1, self.query_shot)
        scores = scores.max(axis=2)[0]

        proposal_deltas = self.bbox_pred(features)
        return scores, proposal_deltas

    def cross_entropy(self, scores, gt_classes, reduction="mean"):
        keeps  = torch.nonzero(gt_classes)
        scores = scores[keeps[:, 0]]
        return cross_entropy(scores, keeps[:, 1], reduction=reduction)

    def losses(self, predictions, proposals, query_ids, keeps=None):
        scores, proposal_deltas = predictions

        # parse classification outputs
        gt_classes = (
            cat([
                p.gt_classes for p in proposals
            ], dim=0) if len(proposals) else torch.empty(0)
        )
        query_ids   = query_ids.reshape(-1, self.query_shot)[:, 0]
        _gt_classes = gt_classes.reshape(-1, 1) == query_ids.reshape(1, -1)
        _gt_classes = _gt_classes.float()
        if keeps is not None:
            _gt_classes = _gt_classes[keeps]
        # _log_classification_stats(scores, gt_classes)

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            gt_boxes = cat([
                (
                    p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes
                ).tensor for p in proposals
            ], dim=0)
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        loss_cls = self.cross_entropy(scores, _gt_classes, reduction="mean")
        loss_reg = self.box_reg_loss(proposal_boxes, gt_boxes, proposal_deltas, gt_classes)
        losses = {
            "loss_cls": loss_cls,
            "loss_box_reg": loss_reg,
        }
        return {
            k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()
        }

    def predict_probs(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        # A little hack to make it work with original inference code
        scores, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]
        probs = nn.functional.softmax(scores, dim=1)
        zeros = torch.zeros((probs.shape[0], 1), device=probs.device)
        probs = torch.cat((probs, zeros), dim=1)
        return probs.split(num_inst_per_image, dim=0)


@ROI_HEADS_REGISTRY.register()
class ROIHeadsQuery(StandardROIHeads):
    def _forward_box(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        rois: torch.Tensor,
    ):
        features = [features[f] for f in self.box_in_features]
        bboxes   = [Boxes(rois[i:i + 1]) for i in range(rois.shape[0])]

        box_features = self.box_pooler(features, bboxes)
        box_features = self.box_head(box_features)
        box_features = self.box_predictor(box_features)
        return box_features

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        outputs = StandardROIHeads._init_box_head(cfg, input_shape)
        outputs["box_predictor"] = FastRCNNOutputLayersQuery(
            cfg, outputs["box_head"].output_shape
        )
        return outputs

    @classmethod
    def _init_mask_head(cls, cfg, input_shape):
        return {}

    def forward(
        self,
        images: ImageList,
        bboxes: torch.Tensor,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        normals: Optional[torch.Tensor] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        return self._forward_box(images, features, bboxes)


class FastRCNNOutputLayersQuery(FastRCNNOutputLayersTarget):
    def forward(self, features):
        if features.dim() > 2:
            features = torch.flatten(features, start_dim=1)
        return self.cls_score(features)


def build_key_heads(cfg, input_shape):
    name = f"{cfg.MODEL.ROI_HEADS.NAME}_Key"
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)
