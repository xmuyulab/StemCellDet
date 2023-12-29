from re import S
import torch
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin
import torch.nn as nn
import torch
import clip
import time
from mmcv.ops.roi_align import roi_align
# from pytorch_memlab import profile,MemReporter
import os
# from PIL import Image
# from mmcv.runner import auto_fp16
from .class_name import *
import time
import torch.nn.functional as F
from torch import distributed as dist
from .visualize import visualize_oam_boxes
from .zip import ZipBackend
import io
import mmcv
from torchvision.transforms import ToPILImage
import numpy as np
import os.path as osp
from PIL import Image
import random
from lvis import LVIS
from mmdet.core import (bbox2roi, bbox_mapping, merge_aug_bboxes,
                        merge_aug_masks, multiclass_nms)
from multiprocessing import Process
from tqdm import tqdm
from .cascade_roi_head_organ_base import CascadeRoIHeadOrganBase

@HEADS.register_module()
class CascadeRoIHeadOrgan(CascadeRoIHeadOrganBase):
    """Simplest base roi head including one bbox head and one mask head."""

    def __init__(self,
                 num_stages,
                 stage_loss_weights,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 load_feature=True,
                 use_clip_inference=False,
                 kd_weight = 256,
                 fixed_lambda=None,
                 prompt_path=None,
                 coco_setting=False,
                 fix_bg=False,
                 ):
        super(CascadeRoIHeadOrgan, self).__init__(num_stages=num_stages,
                                                stage_loss_weights=stage_loss_weights,
                                                bbox_roi_extractor=bbox_roi_extractor,
                                                bbox_head=bbox_head,
                                                mask_roi_extractor=mask_roi_extractor,
                                                mask_head=mask_head,
                                                shared_head=shared_head,
                                                train_cfg=train_cfg,
                                                test_cfg=test_cfg,
                                                load_feature=load_feature,
                                                use_clip_inference=use_clip_inference,
                                                kd_weight = kd_weight,
                                                fixed_lambda=fixed_lambda,
                                                prompt_path=prompt_path,
                                                coco_setting=coco_setting,
                                                fix_bg=fix_bg,
                                              )
        self.project_for_image_clip = nn.Parameter(torch.randn(512, 512), requires_grad=True)
        nn.init.xavier_uniform_(self.project_for_image_clip)
        self.project_for_image_clip.requires_grad = False

        self.attn = nn.MultiheadAttention(embed_dim=512, num_heads=4)
        # self.project_for_image_attn = nn.Identity()
        self.project_for_image_attn = nn.Linear(512, 512)
        nn.init.xavier_uniform_(self.project_for_image_attn.weight)


    def forward_train(self,
                      x,
                      img,
                      img_no_normalize,
                      img_metas,
                      proposal_list,
                      proposals_pre_computed,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        # assign gts and sample proposals
        losses = dict()
        for j in range(self.num_stages):
            self.current_stage = j
            rcnn_train_cfg = self.train_cfg[j]
            lw = self.stage_loss_weights[j]

            if self.with_bbox or self.with_mask:
                bbox_assigner = self.bbox_assigner[j]
                bbox_sampler = self.bbox_sampler[j]
                num_imgs = len(img_metas)
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]
                sampling_results = []
                for i in range(num_imgs):
                    assign_result = bbox_assigner.assign(
                        proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                        gt_labels[i])
                    sampling_result = bbox_sampler.sample(
                        assign_result,
                        proposal_list[i],
                        gt_bboxes[i],
                        gt_labels[i],
                        feats=[lvl_feat[i][None] for lvl_feat in x])
                    sampling_results.append(sampling_result)

            # bbox head forward and loss
            bbox_results = self._bbox_forward_train(j,x,img,img_no_normalize,sampling_results,proposals_pre_computed,
                                                    gt_bboxes, gt_labels,
                                                    img_metas,bbox_assigner,rcnn_train_cfg)

            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{j}.{name}'] = (
                    value * lw if 'loss' in name else value)

            # mask head forward and loss
            if self.with_mask:
                mask_results = self._mask_forward_train(
                j, x, sampling_results, gt_masks, rcnn_train_cfg,
                    bbox_results['bbox_feats'])
                for name, value in mask_results['loss_mask'].items():
                    losses[f's{j}.{name}'] = (
                        value * lw if 'loss' in name else value)

            # refine bboxes
            if j < self.num_stages - 1:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                # bbox_targets is a tuple
                roi_labels = bbox_results['bbox_targets'][0]
                with torch.no_grad():
                    roi_labels = torch.where(
                        roi_labels == self.bbox_head[j].num_classes,
                        bbox_results['cls_score'][:, :-1].argmax(1),
                        roi_labels)
                    proposal_list = self.bbox_head[j].refine_bboxes(
                        bbox_results['rois'], roi_labels,
                        bbox_results['bbox_pred'], pos_is_gts, img_metas)
                losses.update(bbox_results['loss_bbox'])
        return losses


    def _bbox_forward_train(self, stage, x, img, img_no_normalize, sampling_results, proposals_pre_computed, gt_bboxes, gt_labels,
                            img_metas,bbox_assigner,train_cfg):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        if not self.fix_bg:
            input_one = x[0].new_ones(1)
            bg_class_embedding = self.bg_embedding(input_one).reshape(1, 512)
            bg_class_embedding = torch.nn.functional.normalize(bg_class_embedding, p=2, dim=1)
        # ----------------------------------------------------------
        num_proposals_per_img = tuple(len(proposal) for proposal in proposals_pre_computed)
        rois_image = torch.cat(proposals_pre_computed, dim=0)
        batch_index = torch.cat([x[0].new_full((num_proposals_per_img[i],1),i) for i in range(len(num_proposals_per_img))],0)
        rois_image = torch.cat([batch_index, rois_image[..., :4]], dim=-1)
        bboxes = rois_image
        bbox_results, region_embeddings = self._bbox_forward(stage, x, rois)
        bbox_targets = self.bbox_head[stage].get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, train_cfg)
        labels, _, _, _ = bbox_targets
        
        region_embeddings = self.projection[stage](region_embeddings)
        # region_embeddings = torch.nn.functional.normalize(region_embeddings, p=2, dim=1)
        cls_score_text = self.project_for_cls[stage](region_embeddings)
        if not self.fix_bg:
            text_features = torch.cat([self.text_features_for_classes, bg_class_embedding], dim=0)
        else:
            text_features = self.text_features_for_classes

        # cls_score_text = region_embeddings @ text_features.T
        # text_cls_loss = F.cross_entropy(cls_score_text / self.temperature, labels, reduction='mean')
        text_cls_loss = F.cross_entropy(cls_score_text, labels, reduction='mean')

        loss_bbox = self.bbox_head[stage].loss(
            bbox_results['bbox_pred'], rois,
            *bbox_targets)
        loss_bbox.update(text_cls_loss=text_cls_loss)
        bbox_results.update(loss_bbox=loss_bbox, cls_score=cls_score_text, rois=rois, bbox_targets=bbox_targets)
        return bbox_results

    def simple_test(self,
                    x,
                    img,
                    img_no_normalize,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False,
                    **kwargs):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        num_imgs = len(proposal_list)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg
        if not self.fix_bg:
            input_one = x[0].new_ones(1)
            bg_class_embedding = self.bg_embedding(input_one).unsqueeze(0)
            bg_class_embedding = torch.nn.functional.normalize(bg_class_embedding, p=2, dim=1)
            text_features = torch.cat([self.text_features_for_classes, bg_class_embedding], dim=0)
        else:
            text_features = self.text_features_for_classes

        rois = bbox2roi(proposal_list)
        for i in range(self.num_stages):
            bbox_results, region_embeddings = self._bbox_forward(i, x, rois)
            region_embeddings = self.projection[i](region_embeddings)
            # region_embeddings = torch.nn.functional.normalize(region_embeddings, p=2, dim=1)
            cls_score_text = self.project_for_cls[i](region_embeddings)

            # cls_score_text = region_embeddings @ text_features.T
            # cls_score_text = cls_score_text / 0.007
            cls_score_text = cls_score_text.softmax(dim=1)

            # only use cls branch
            cls_score = cls_score_text

            # split batch bbox prediction back to each image
            # cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']
            num_proposals_per_img = tuple(
                len(proposals) for proposals in proposal_list)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head[i].bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
            ms_scores.append(cls_score)

            if i < self.num_stages - 1:
                bbox_label = [s[:, :-1].argmax(dim=1) for s in cls_score]
                rois = torch.cat([
                    self.bbox_head[i].regress_by_class(rois[j], bbox_label[j],
                                                       bbox_pred[j],
                                                       img_metas[j])
                    for j in range(num_imgs)
                ])

        # average scores of each image by stages
        cls_score = [
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(num_imgs)
        ]

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(num_imgs):
            # print((cls_score[i]>0.001).sum())
            det_bbox, det_label = self.bbox_head[-1].get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        if torch.onnx.is_in_onnx_export():
            return det_bboxes, det_labels
        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head[-1].num_classes)
            for i in range(num_imgs)
        ]
        ms_bbox_result['ensemble'] = bbox_results

        if self.with_mask:
            if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)]
                                for _ in range(num_imgs)]
            else:
                if rescale and not isinstance(scale_factors[0], float):
                    scale_factors = [
                        torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                        for scale_factor in scale_factors
                    ]
                _bboxes = [
                    det_bboxes[i][:, :4] *
                    scale_factors[i] if rescale else det_bboxes[i][:, :4]
                    for i in range(len(det_bboxes))
                ]
                mask_rois = bbox2roi(_bboxes)
                num_mask_rois_per_img = tuple(
                    _bbox.size(0) for _bbox in _bboxes)
                aug_masks = []
                for i in range(self.num_stages):
                    mask_results = self._mask_forward(i, x, mask_rois)
                    mask_pred = mask_results['mask_pred']
                    # split batch mask prediction back to each image
                    mask_pred = mask_pred.split(num_mask_rois_per_img, 0)
                    aug_masks.append(
                        [m.sigmoid().cpu().numpy() for m in mask_pred])

                # apply mask post-processing to each image individually
                segm_results = []
                for i in range(num_imgs):
                    if det_bboxes[i].shape[0] == 0:
                        segm_results.append(
                            [[]
                             for _ in range(self.mask_head[-1].num_classes)])
                    else:
                        aug_mask = [mask[i] for mask in aug_masks]
                        merged_masks = merge_aug_masks(
                            aug_mask, [[img_metas[i]]] * self.num_stages,
                            rcnn_test_cfg)
                        segm_result = self.mask_head[-1].get_seg_masks(
                            merged_masks, _bboxes[i], det_labels[i],
                            rcnn_test_cfg, ori_shapes[i], scale_factors[i],
                            rescale)
                        segm_results.append(segm_result)
            ms_segm_result['ensemble'] = segm_results

        if self.with_mask:
            results = list(
                zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble']))
        else:
            results = ms_bbox_result['ensemble']

        return results

    # # useless
    # def simple_test_bboxes(self,
    #                        x,
    #                        img,
    #                        img_no_normalize,
    #                        img_metas,
    #                        proposals,
    #                        proposals_pre_computed,
    #                        rcnn_test_cfg,
    #                        rescale=False):
    #     # get origin input shape to support onnx dynamic input shape
    #     img_shapes = tuple(meta['img_shape'] for meta in img_metas)
    #     scale_factors = tuple(meta['scale_factor'] for meta in img_metas)
    #     rois = bbox2roi(proposals)
    #     num_proposals_per_img = tuple(len(proposal) for proposal in proposals)
    #
    #     bbox_results, region_embeddings = self._bbox_forward(x,rois)
    #     region_embeddings = self.projection(region_embeddings)
    #     region_embeddings = torch.nn.functional.normalize(region_embeddings,p=2,dim=1)
    #     if not self.fix_bg:
    #         input_one = x[0].new_ones(1)
    #         bg_class_embedding = self.bg_embedding(input_one).unsqueeze(0)
    #         bg_class_embedding = torch.nn.functional.normalize(bg_class_embedding,p=2,dim=1)
    #         text_features = torch.cat([self.text_features_for_classes,bg_class_embedding],dim=0)
    #     else:
    #         text_features = self.text_features_for_classes
    #     #-----------------------------------------------------
    #     cls_score_text = region_embeddings@text_features.T
    #     cls_score = cls_score_text
    #     # """
    #     bbox_pred = bbox_results['bbox_pred']
    #     num_proposals_per_img = tuple(len(p) for p in proposals)
    #     rois = rois.split(num_proposals_per_img, 0)
    #     cls_score = cls_score.split(num_proposals_per_img, 0)
    #
    #     # some detector with_reg is False, bbox_pred will be None
    #     if bbox_pred is not None:
    #         # the bbox prediction of some detectors like SABL is not Tensor
    #         if isinstance(bbox_pred, torch.Tensor):
    #             bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
    #         else:
    #             bbox_pred = self.bbox_head.bbox_pred_split(
    #                 bbox_pred, num_proposals_per_img)
    #     else:
    #         bbox_pred = (None,) * len(proposals)
    #
    #     # apply bbox post-processing to each image individually
    #     det_bboxes = []
    #     det_labels = []
    #     for i in range(len(proposals)):
    #         det_bbox, det_label = self.bbox_head.get_bboxes(
    #             rois[i],
    #             cls_score[i],
    #             bbox_pred[i],
    #             img_shapes[i],
    #             scale_factors[i],
    #             rescale=rescale,
    #             cfg=rcnn_test_cfg)
    #         det_bboxes.append(det_bbox)
    #         det_labels.append(det_label)
    #     return det_bboxes, det_labels