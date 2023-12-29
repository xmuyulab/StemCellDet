import copy
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
from tqdm import tqdm
from .standard_roi_head_organ_base import StandardRoIHeadOrganBase

@HEADS.register_module()
class StandardRoIHeadOrgan(StandardRoIHeadOrganBase):
    """Simplest base roi head including one bbox head and one mask head."""

    def __init__(self,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 load_feature=True,
                 use_clip_inference=False,
                 kd_weight=256,
                 fixed_lambda=None,
                 prompt_path=None,
                 coco_setting=False,
                 fix_bg=False,
                 feature_path='data/lvis_clip_image_embedding.zip'
                 ):
        super(StandardRoIHeadOrgan, self).__init__(bbox_roi_extractor=bbox_roi_extractor,
                                                   bbox_head=bbox_head,
                                                   mask_roi_extractor=mask_roi_extractor,
                                                   mask_head=mask_head,
                                                   shared_head=shared_head,
                                                   train_cfg=train_cfg,
                                                   test_cfg=test_cfg,
                                                   load_feature=load_feature,
                                                   use_clip_inference=use_clip_inference,
                                                   kd_weight=kd_weight,
                                                   fixed_lambda=fixed_lambda,
                                                   prompt_path=prompt_path,
                                                   coco_setting=coco_setting,
                                                   fix_bg=fix_bg,
                                                   feature_path=feature_path,
                                              )
        self.project_for_image_clip = nn.Parameter(torch.randn(512, 512), requires_grad=True)
        nn.init.xavier_uniform_(self.project_for_image_clip)
        # self.project_for_image_clip.requires_grad = False

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

        proposals_pre_computed_pos = []
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)
                # L10 : add to sample pos rpn
                gt_bboxes_add_score = torch.ones(gt_labels[i].shape[0], 5, device=self.device)
                gt_bboxes_add_score[:, :4] = gt_bboxes[i].data
                gt_bboxes_add_score = torch.cat([gt_bboxes_add_score, proposal_list[i].data], dim=0)
                proposals_pre_computed_pos.append(gt_bboxes_add_score[sampling_result.pos_inds.data.cpu()])
                # inds = torch.cat([sampling_result.pos_inds.data.cpu(), sampling_result.neg_inds.data.cpu()], dim=0)[:16]
                # proposals_pre_computed_pos.append(proposals_pre_computed[i][inds])

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, img, img_no_normalize, sampling_results, proposals_pre_computed_pos,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        return losses

    def _bbox_forward_train(self, x, img, img_no_normalize, sampling_results, proposals_pre_computed, gt_bboxes,
                            gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        if not self.fix_bg:
            input_one = x[0].new_ones(1)
            bg_class_embedding = self.bg_embedding(input_one).reshape(1, 512)
            bg_class_embedding = torch.nn.functional.normalize(bg_class_embedding, p=2, dim=1)
        # ----------------------------------------------------------
        num_proposals_per_img = tuple(len(proposal) for proposal in proposals_pre_computed)
        rois_image = torch.cat(proposals_pre_computed, dim=0)
        batch_index = torch.cat(
            [x[0].new_full((num_proposals_per_img[i], 1), i) for i in range(len(num_proposals_per_img))], 0)
        rois_image = torch.cat([batch_index, rois_image[..., :4]], dim=-1)
        bboxes = rois_image
        # ------------------------------------------------------------
        # not using precomputed proposals
        bbox_results, region_embeddings = self._bbox_forward(x, rois)
        # # -------------------------------------------------------------
        if self.ensemble:
            _, region_embeddings_image = self._bbox_forward_for_image(x, bboxes)
            region_embeddings_image = self.projection_for_image(region_embeddings_image)
            region_embeddings_image = torch.nn.functional.normalize(region_embeddings_image, p=2, dim=1)
        else:
            _, region_embeddings_image = self._bbox_forward(x, bboxes)
            region_embeddings_image = self.projection(region_embeddings_image)
            region_embeddings_image = torch.nn.functional.normalize(region_embeddings_image, p=2, dim=1)

        if self.load_feature:
            clip_image_features_ensemble = []
            bboxes_all = bboxes.split(num_proposals_per_img)
            for i in range(len(img_metas)):
                save_path = '.pth'
                try:
                    f = self.zipfile.get(save_path)
                    stream = io.BytesIO(f)
                    tmp = torch.load(stream)
                    clip_image_features_ensemble.append(tmp.to(self.device))
                except:
                    bboxes_single_image = bboxes_all[i]
                    # bboxes15 = self.boxto15(bboxes_single_image)
                    # self.checkdir(save_path)
                    clip_image_features = self.img2pil2feat(img_no_normalize[i], bboxes_single_image[:, 1:])
                    # clip_image_features15 = self.img2pil2feat(img_no_normalize[i], bboxes15[:, 1:])
                    clip_image_features_single = clip_image_features
                    # clip_image_features_single = clip_image_features + clip_image_features15
                    clip_image_features_single = clip_image_features_single.float()
                    clip_image_features_single = torch.nn.functional.normalize(clip_image_features_single, p=2, dim=1)
                    # torch.save(clip_image_features_single.cpu(), save_path)
                    clip_image_features_ensemble.append(clip_image_features_single)
            clip_image_features_ensemble = torch.cat(clip_image_features_ensemble, dim=0)
        else:
            clip_image_features_ensemble = []
            bboxes_all = bboxes.split(num_proposals_per_img)
            for i in range(len(img_metas)):
                bboxes_single_image = bboxes_all[i]
                # bboxes15 = self.boxto15(bboxes_single_image)
                # self.checkdir(save_path)
                clip_image_features = self.img2pil2feat(img_no_normalize[i], bboxes_single_image[:, 1:])
                # clip_image_features15 = self.img2pil2feat(img_no_normalize[i], bboxes15[:, 1:])
                clip_image_features_single = clip_image_features
                # clip_image_features_single = clip_image_features + clip_image_features15
                clip_image_features_single = clip_image_features_single.float()
                clip_image_features_single = torch.nn.functional.normalize(clip_image_features_single, p=2, dim=1)
                clip_image_features_ensemble.append(clip_image_features_single)
                # torch.save(clip_image_features_single.cpu(), save_path)
            clip_image_features_ensemble = torch.cat(clip_image_features_ensemble, dim=0)
            # clip_image_features_ensemble_align = torch.cat(clip_image_features_ensemble_align, dim=0)
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        labels, _, _, _ = bbox_targets

        region_embeddings = self.projection(region_embeddings)
        region_embeddings = torch.nn.functional.normalize(region_embeddings, p=2, dim=1)
        if not self.fix_bg:
            text_features = torch.cat([self.text_features_for_classes, bg_class_embedding], dim=0)
        else:
            text_features = self.text_features_for_classes

        # text_features = F.normalize(text_features, p=2, dim=1)
        cls_score_text = region_embeddings @ text_features.T
        cls_score_text[:, self.novel_label_ids] = -1e11
        # text_cls_loss = F.cross_entropy(cls_score_text / self.temperature, labels,
        #                                 weight=torch.Tensor([1.0, 3.0, 1.0, 1.0]).cuda(), reduction='mean')
        text_cls_loss = F.cross_entropy(cls_score_text / self.temperature, labels, reduction='mean')
        kd_loss = F.l1_loss(region_embeddings_image, clip_image_features_ensemble)
        loss_bbox = self.bbox_head.loss(
            bbox_results['bbox_pred'], rois,
            *bbox_targets)
        # loss_bbox.update(text_cls_loss=text_cls_loss)
        loss_bbox.update(text_cls_loss=text_cls_loss, kd_loss=kd_loss * self.kd_weight)
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def simple_test_bboxes(self,
                           x,
                           img,
                           img_no_normalize,
                           img_metas,
                           proposals,
                           proposals_pre_computed,
                           rcnn_test_cfg,
                           rescale=False):

        # get origin input shape to support onnx dynamic input shape
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)
        # if self.use_clip_inference:
        rois = bbox2roi(proposals)
        num_proposals_per_img = tuple(len(proposal) for proposal in proposals)
        bbox_results, region_embeddings = self._bbox_forward(x, rois)
        region_embeddings = self.projection(region_embeddings)
        region_embeddings = torch.nn.functional.normalize(region_embeddings, p=2, dim=1)
        if not self.fix_bg:
            input_one = x[0].new_ones(1)
            bg_class_embedding = self.bg_embedding(input_one).unsqueeze(0)
            bg_class_embedding = torch.nn.functional.normalize(bg_class_embedding, p=2, dim=1)
            text_features = torch.cat([self.text_features_for_classes, bg_class_embedding], dim=0)
        else:
            text_features = self.text_features_for_classes
        # -----------------------------------------------------
        # text_features = F.normalize(text_features, p=2, dim=1)
        cls_score_text = region_embeddings @ text_features.T

        if self.num_classes == 80 and self.coco_setting:
            cls_score_text[:, self.unseen_label_ids_test] = -1e11
        cls_score_text = cls_score_text / 0.007
        cls_score_text = cls_score_text.softmax(dim=1)
        # --------------------------------------------
        if self.ensemble and not self.use_clip_inference:
            _, region_embeddings_image = self._bbox_forward_for_image(x, rois)
            region_embeddings_image = self.projection_for_image(region_embeddings_image)
            region_embeddings_image = torch.nn.functional.normalize(region_embeddings_image, p=2, dim=1)
            cls_score_image = region_embeddings_image @ text_features.T
            cls_score_image = cls_score_image / 0.007
            if self.num_classes == 80 and self.coco_setting:
                cls_score_image[:, self.unseen_label_ids_test] = -1e11
            cls_score_image[:, -1] = -1e11
            cls_score_image = cls_score_image.softmax(dim=1)
        # ------------------------------------------------
        # using clip to inference
        if self.ensemble and self.use_clip_inference:
            bboxes = rois
            save_path = os.path.join('./data/lvis_clip_image_embedding_test_offline',
                                     img_metas[0]['ori_filename'].split('.')[0] + '.pth')
            # save_path = os.path.join('./data/lvis_clip_image_embedding_test_offline_img2pil', img_metas[0]['ori_filename'].split('.')[0] + '.pth')
            if not osp.exists(save_path):
                # if True:
                # bboxes15 = self.boxto15(bboxes)
                clip_image_features_align = self.clip_image_forward_align(img, bboxes, num_proposals_per_img)
                # clip_image_features15_align = self.clip_image_forward_align(img, bboxes15, num_proposals_per_img)
                # clip_image_features_ensemble_align = clip_image_features_align + clip_image_features15_align
                clip_image_features_ensemble_align = clip_image_features_align
                clip_image_features_ensemble_align = clip_image_features_ensemble_align.float()
                clip_image_features_ensemble = F.normalize(clip_image_features_ensemble_align, p=2, dim=1)
                # torch.save(clip_image_features_ensemble_img2pil.cpu(), save_path)
                # self.checkdir(save_path)
                # torch.save(clip_image_features_ensemble.cpu(), save_path)
            else:
                clip_image_features_ensemble = torch.load(save_path).to(self.device)
                # clip_image_features_ensemble_img2pil = torch.load(save_path).to(self.device)
            cls_score_clip = clip_image_features_ensemble @ text_features.T
            cls_score_clip = cls_score_clip / 0.007
            # cls_score_clip[:, :-1] = cls_score_clip[:, :-1] / cls_score_clip[:, :-1].std(dim=1, keepdim=True) * 4
            cls_score_clip[:, -1] = -1e11
            if self.num_classes == 80 and self.coco_setting:
                cls_score_clip[:, self.unseen_label_ids_test] = -1e11
            cls_score_clip = cls_score_clip.softmax(dim=1)
            cls_score_image = cls_score_clip
        # --------------------------------------------------
        # """
        a = 1 / 3
        if self.ensemble:
            if self.fixed_lambda is not None:
                cls_score = cls_score_image ** (1 - self.fixed_lambda) * cls_score_text ** self.fixed_lambda
            else:
                cls_score = torch.where(self.novel_index, cls_score_image ** (1 - a) * cls_score_text ** a,
                                        cls_score_text ** (1 - a) * cls_score_image ** a)
        else:
            cls_score = cls_score_text
        # cls_score[:, ~self.novel_index] *= 0
        # cls_score = cls_score_image
        # """
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None,) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels

    def clip_image_forward_align(self, img, bboxes, num_proposals_per_img, flag=False):
        cropped_images = roi_align(img, bboxes, (224, 224))
        image_features = self.clip_model.encode_image(cropped_images).float()
        image_features = image_features @ self.project_for_image_clip.T
        return image_features

    def get_single_clip_finetune_feature(self, img, bboxes):
        # get all clip finetune feature in single img (may have one more bboxes in one image)
        clip_image_features_single = self.img2pil2feat(img, bboxes)
        clip_image_features_single = clip_image_features_single.float()
        clip_image_features_single = clip_image_features_single @ self.project_for_image_clip.T
        clip_image_features_single = torch.nn.functional.normalize(clip_image_features_single, p=2, dim=1)
        return clip_image_features_single

    def forward_for_clip_finetune_test(self, img_no_normalize, gt_bboxes, gt_labels):
        clip_image_features = []
        for i in range(len(img_no_normalize)):
            clip_image_features_single = self.get_single_clip_finetune_feature(img_no_normalize[i], gt_bboxes[i])
            clip_image_features.append(clip_image_features_single)
        clip_image_features = torch.cat(clip_image_features, dim=0)
        labels = torch.cat(gt_labels, dim=0)

        text_features = self.text_features_for_classes
        cls_score_text = clip_image_features @ text_features.T
        res = dict({
            "pred": cls_score_text.max(dim=-1)[1].data.cpu(),
            "gdth": labels.data.cpu()
        })
        return [res]

    def forward_for_clip_finetune(self, img_no_normalize, gt_bboxes, gt_labels):
        clip_image_features = []
        for i in range(len(img_no_normalize)):
            clip_image_features_single = self.get_single_clip_finetune_feature(img_no_normalize[i], gt_bboxes[i])
            clip_image_features.append(clip_image_features_single)
        clip_image_features = torch.cat(clip_image_features, dim=0)
        labels = torch.cat(gt_labels, dim=0)

        text_features = self.text_features_for_classes
        cls_score_text = clip_image_features @ text_features.T
        cls_score_text[:, self.novel_label_ids] = -1e11
        text_cls_loss = F.cross_entropy(cls_score_text / self.temperature, labels, weight=torch.Tensor([1.0, 3.0, 1.0]),
                                        reduction='mean')
        return dict({"clip_finetune_loss": text_cls_loss})