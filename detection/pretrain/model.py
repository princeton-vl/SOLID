# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# This file contains code under Apache-2.0 License from https://github.com/facebookresearch/detectron2
import torch
import torch.nn as nn
import numpy as np

from typing import Dict, List, Union, Optional, Tuple

from detectron2.config import configurable
from detectron2.modeling import GeneralizedRCNN, META_ARCH_REGISTRY, Backbone
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.data.detection_utils import convert_image_to_rgb

from pretrain.utils import concat_all_gather, unique
from pretrain.roi_heads import build_key_heads


@META_ARCH_REGISTRY.register()
class Wrapper(GeneralizedRCNN):
    @configurable
    def __init__(
        self,
        *,
        backbone_q: Backbone,
        proposal_generator_q: nn.Module,
        roi_heads_q: nn.Module,
        backbone_k: Backbone,
        proposal_generator_k: nn.Module,
        roi_heads_k: nn.Module,
        num_classes: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        momentum: float = 0.999,
        output_dir: str = None,
        embd_dims: List[int] = [256],
        eman: bool = False,
    ):
        super(GeneralizedRCNN, self).__init__()
        self.backbone            = backbone_q
        self.proposal_generator  = proposal_generator_q
        self.roi_heads           = roi_heads_q
        self.backbone_k          = backbone_k
        self.roi_heads_k         = roi_heads_k
        self.output_dir          = output_dir
        self.momentum            = momentum
        self.eman                = eman

        self.input_format = input_format
        self.vis_period   = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shape"

        self.num_cascade_stages = 1
        if hasattr(self.roi_heads, "num_cascade_stages"):
            self.num_cascade_stages = self.roi_heads.num_cascade_stages

        self.register_buffer("queue", torch.randn(num_classes, embd_dims[-1] * self.num_cascade_stages))
        self.register_buffer("queue_ids", torch.arange(num_classes, dtype=torch.long))

        self._copy_and_freeze_params(backbone_q, backbone_k)
        self._copy_and_freeze_params(roi_heads_q, roi_heads_k)

    def _preprocess_image(self, batched_inputs: List[torch.Tensor]) -> ImageList:
        images = [x.to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        return images

    def _copy_and_freeze_params(self, module_q, module_k):
        for param_q, param_k in zip(module_q.parameters(), module_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self, module_q, module_k):
        if self.eman:
            state_dict_q = module_q.state_dict()
            state_dict_k = module_k.state_dict()
            for (q_k, q_v), (k_k, k_v) in zip(state_dict_q.items(), state_dict_k.items()):
                if 'num_batches_tracked' in k_k:
                    k_v.copy_(q_v)
                    continue
                k_v.copy_(k_v * self.momentum + q_v * (1. - self.momentum))
            return

        for param_q, param_k in zip(module_q.parameters(), module_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _momentum_update_wrapper(self):
        self._momentum_update(self.backbone, self.backbone_k)
        self._momentum_update(self.roi_heads, self.roi_heads_k)

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
            batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not self.training:
            return self.inference(batched_inputs)

        self._momentum_update_wrapper()

        images  = [b["image"] for b in batched_inputs]
        queries = batched_inputs[0]["queries"]
        query_ids    = batched_inputs[0]["query_ids"]
        query_bboxes = batched_inputs[0]["query_bboxes"]

        images  = ImageList.from_tensors(
            self._preprocess_image(images),
            self.backbone.size_divisibility,
        )
        queries = ImageList.from_tensors(
            self._preprocess_image(queries),
            self.backbone_k.size_divisibility,
        )

        gt_instances: Union[List[torch.Tensor], None] = None
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        gt_normals: Union[torch.Tensor, None] = None
        if "normal" in batched_inputs[0]:
            gt_normals = [x["normal"].to(self.device) for x in batched_inputs]
            gt_normals = ImageList.from_tensors(gt_normals, self.backbone.size_divisibility).tensor

        features_q = self.backbone(images.tensor)

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(
                images,
                features_q,
                gt_instances,
            )
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        with torch.no_grad():
            features_k = self.backbone_k(queries.tensor)
            features_k = self.roi_heads_k(
                queries.tensor,
                query_bboxes,
                features_k,
                proposals,
                gt_instances,
                None
            )

        query_ids  = concat_all_gather(query_ids)
        features_k = concat_all_gather(features_k)

        query_ids, uniq_ids = unique(query_ids, dim=0)
        features_k = features_k[uniq_ids]
        self.queue[query_ids] = features_k.detach().to(self.queue.dtype)

        features_q["queries"]   = self.queue.clone().detach().to(features_k.dtype)
        features_q["query_ids"] = self.queue_ids.clone().detach()

        _, detector_losses = self.roi_heads(
            images,
            features_q,
            proposals,
            gt_instances,
        )
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(
                    batched_inputs,
                    proposals,
                )

        if "predictions" in detector_losses:
            del detector_losses["predictions"]
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        assert not self.training

        images  = [b["image"] for b in batched_inputs]
        queries = batched_inputs[0]["queries"]
        query_ids    = batched_inputs[0]["query_ids"]
        query_bboxes = batched_inputs[0]["query_bboxes"]

        images  = ImageList.from_tensors(
            self._preprocess_image(images),
            self.backbone.size_divisibility,
        )
        queries = ImageList.from_tensors(
            self._preprocess_image(queries),
            self.backbone_k.size_divisibility,
        )

        features_q = self.backbone(images.tensor)
        if not hasattr(self, "features_k"):
            features_k_list = []
            batch_size   = 64
            query_bboxes = query_bboxes.cuda()
            for i in range((queries.tensor.shape[0] + batch_size - 1) // batch_size):
                sind = i * batch_size
                eind = min(sind + batch_size, queries.tensor.shape[0])

                queries_tensor = queries.tensor[sind:eind]
                q_bboxes = query_bboxes[sind:eind]

                features_k = self.backbone_k(queries_tensor)
                features_k = self.roi_heads_k(
                    queries_tensor,
                    q_bboxes,
                    features_k,
                    None,
                    None,
                    None
                )
                features_k_list.append(features_k)
            self.features_k = torch.cat(features_k_list)

        features_q["queries"]   = self.features_k
        features_q["query_ids"] = query_ids

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features_q, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            results, _ = self.roi_heads(images, features_q, proposals, None, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features_q, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            results   = GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
            proposals = GeneralizedRCNN._postprocess(proposals, batched_inputs, images.image_sizes)
            for result, proposal in zip(results, proposals):
                result["proposals"] = proposal["instances"]
            return results
        else:
            return results

    def train(self, mode=True):
        super(Wrapper, self).train(mode=mode)
        if self.eman:
            self.backbone_k.eval()
            self.roi_heads_k.eval()
        return self

    @classmethod
    def from_config(cls, cfg):
        backbone_q = build_backbone(cfg)
        backbone_k = build_backbone(cfg)

        proposal_generator_q = build_proposal_generator(cfg, backbone_q.output_shape())
        proposal_generator_k = build_proposal_generator(cfg, backbone_k.output_shape())

        roi_heads_q = build_roi_heads(cfg, backbone_q.output_shape())
        roi_heads_k = build_key_heads(cfg, backbone_k.output_shape())

        return {
            "backbone_q": backbone_q,
            "proposal_generator_q": proposal_generator_q,
            "roi_heads_q": roi_heads_q,
            "backbone_k": backbone_k,
            "proposal_generator_k": proposal_generator_k,
            "roi_heads_k": roi_heads_k,
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "momentum": cfg.MODEL.MOMENTUM,
            "output_dir": cfg.OUTPUT_DIR,
            "embd_dims": cfg.MODEL.ROI_HEADS.EMBD_DIMS,
        }
