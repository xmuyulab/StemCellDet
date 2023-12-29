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
from collections import OrderedDict
from ..gradient_reversal import GradientReversal

from .triplet_loss import TripletLoss
from .clip_lora import clip_lora
import torchvision

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

        self.project_lora_a = nn.Linear(512, 8, bias=False)
        nn.init.normal_(self.project_lora_a.weight)
        self.project_lora_b = nn.Linear(8, 512, bias=False)
        nn.init.constant_(self.project_lora_b.weight, 0)
        # self.project_for_image_clip.requires_grad = False

        self.attn = nn.MultiheadAttention(embed_dim=512, num_heads=4)
        # self.project_for_image_attn = nn.Identity()
        self.project_for_image_attn = nn.Linear(512, 512)
        nn.init.xavier_uniform_(self.project_for_image_attn.weight)
        # use for onehot cls
        self.project_for_cls = nn.Linear(512, 3+1)
        nn.init.xavier_uniform_(self.project_for_cls.weight)
        # use for multi
        self.project_for_multi = nn.Linear(512, 512)
        nn.init.xavier_uniform_(self.project_for_multi.weight)

        self.clip_lora = clip_lora()
        self.resnet_grad = resnet_grad(self.text_features_for_classes)


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
                # proposals_pre_computed_pos.append(gt_bboxes_add_score[:32])

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
            text_features = torch.cat([self.text_features_for_classes, bg_class_embedding], dim=0)
        else:
            text_features = self.text_features_for_classes
        # ------------------------------------------------------------
        # not using precomputed proposals
        bbox_results, region_embeddings = self._bbox_forward(x, rois)
        # # -------------------------------------------------------------
        # num_proposals_per_img = tuple(len(proposal) for proposal in proposals_pre_computed)
        # rois_image = torch.cat(proposals_pre_computed, dim=0)
        # batch_index = torch.cat(
        #     [x[0].new_full((num_proposals_per_img[i], 1), i) for i in range(len(num_proposals_per_img))], 0)
        # bboxes = torch.cat([batch_index, rois_image[..., :4]], dim=-1)
        # if self.ensemble:
        #     _, region_embeddings_image = self._bbox_forward_for_image(x, bboxes)
        #     region_embeddings_image = self.projection_for_image(region_embeddings_image)
        #     region_embeddings_image = torch.nn.functional.normalize(region_embeddings_image, p=2, dim=1)
        # else:
        #     _, region_embeddings_image = self._bbox_forward(x, bboxes)
        #     region_embeddings_image = self.projection(region_embeddings_image)
        #     region_embeddings_image = torch.nn.functional.normalize(region_embeddings_image, p=2, dim=1)
        # # clip finetune part
        # clip_image_features_ensemble = []
        # for i in range(len(img_metas)):
        #     clip_image_features = self.img2pil2feat(img_no_normalize[i], proposals_pre_computed[i][:, :4])
        #     clip_image_features_single = clip_image_features.float() @ self.project_for_image_clip.T
        #     clip_image_features_single = torch.nn.functional.normalize(clip_image_features_single, p=2, dim=1)
        #     clip_image_features_ensemble.append(clip_image_features_single)
        # clip_image_features_ensemble = torch.cat(clip_image_features_ensemble, dim=0)
        ###
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        labels, _, _, _ = bbox_targets

        # cls branch part
        region_embeddings = self.projection(region_embeddings)
        region_embeddings = self.project_for_multi(region_embeddings)
        # region_embeddings = torch.nn.functional.normalize(region_embeddings, p=2, dim=1) # afn

        # # text_embedding_part = self.text_features_for_classes.repeat(region_embeddings.shape[0],1,1)
        # text_embedding_part = self.text_features_for_classes[self.base_label_ids].repeat(region_embeddings.shape[0],1,1)
        # region_embeddings = self.attn(region_embeddings.unsqueeze(dim=0) / 0.1,
        #                               text_embedding_part.permute(1, 0, 2) / 0.1,
        #                               text_embedding_part.permute(1, 0, 2))[0].squeeze(dim=0)
        # region_embeddings = region_embeddings + self.attn(region_embeddings.unsqueeze(dim=0) / 0.1,
        #                               text_embedding_part.permute(1, 0, 2) / 0.1,
        #                               text_embedding_part.permute(1, 0, 2))[0].squeeze(dim=0)
        # region_embeddings = self.attn(text_embedding_part.permute(1, 0, 2) / 0.1,
        #                               region_embeddings.unsqueeze(dim=0) / 0.1,
        #                               region_embeddings.unsqueeze(dim=0))[0].mean(0)
        # region_embeddings = region_embeddings + self.attn(text_embedding_part.permute(1, 0, 2) / 0.1,
        #                               region_embeddings.unsqueeze(dim=0) / 0.1,
        #                               region_embeddings.unsqueeze(dim=0))[0].mean(0)
        # region_embeddings = self.project_for_image_attn(region_embeddings)
        # region_embeddings = torch.nn.functional.normalize(region_embeddings, p=2, dim=1)

        cls_score_text = region_embeddings @ text_features.T
        text_cls_loss = F.cross_entropy(cls_score_text, labels,
                                        weight=torch.Tensor([1.0, 1.0, 1.0, 1.0]).cuda(), reduction='mean')
        # cls_score_text = self.project_for_cls(region_embeddings)
        # text_cls_loss = F.cross_entropy(cls_score_text, labels,
        #                                 weight=torch.Tensor([1.0, 1.0, 1.0, 1.0]).cuda(), reduction='mean')

        # kd_loss = F.l1_loss(region_embeddings_image, clip_image_features_ensemble) * self.kd_weight
        loss_bbox = self.bbox_head.loss(
            bbox_results['bbox_pred'], rois,
            *bbox_targets)
        # loss_bbox.update(text_cls_loss=text_cls_loss, kd_loss=kd_loss)
        loss_bbox.update(text_cls_loss=text_cls_loss)
        bbox_results.update(loss_bbox=loss_bbox)

        # ######
        # # pos_inds_save = pos_inds # all
        # pos_inds_save = sampling_results[0].pos_is_gt # gt
        # if len(pos_inds_save) != 0:
        #     print(img_metas[0]['ori_filename'],
        #           cls_score_text.max(dim=-1)[1][:len(pos_inds_save)][
        #               pos_inds_save].data.cpu().tolist(),
        #           labels[:len(pos_inds_save)][pos_inds_save].data.cpu().tolist())
        #     # torch.save(region_embeddings[:len(pos_inds_save)][
        #     #           pos_inds_save].data.cpu(), os.path.join('./data/tmp',
        #     #                                                   img_metas[0]['ori_filename'].replace('.png', '.pth')))
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
        # cls branch part
        region_embeddings = self.projection(region_embeddings)
        region_embeddings = self.project_for_multi(region_embeddings)
        # region_embeddings = torch.nn.functional.normalize(region_embeddings, p=2, dim=1) # afn

        # text_embedding_part = self.text_features_for_classes.repeat(region_embeddings.shape[0],1,1)
        # region_embeddings = self.attn(text_embedding_part.permute(1, 0, 2) / 0.1,
        #                               region_embeddings.unsqueeze(dim=0) / 0.1,
        #                               region_embeddings.unsqueeze(dim=0))[0].mean(0)
        # region_embeddings = self.project_for_image_attn(region_embeddings)
        # region_embeddings = torch.nn.functional.normalize(region_embeddings, p=2, dim=1)
        if not self.fix_bg:
            input_one = x[0].new_ones(1)
            bg_class_embedding = self.bg_embedding(input_one).unsqueeze(0)
            bg_class_embedding = torch.nn.functional.normalize(bg_class_embedding, p=2, dim=1)
            text_features = torch.cat([self.text_features_for_classes, bg_class_embedding], dim=0)
        else:
            text_features = self.text_features_for_classes
        # -----------------------------------------------------
        cls_score_text = region_embeddings @ text_features.T
        # cls_score_text = self.project_for_cls(region_embeddings)
        # cls_score_text = cls_score_text / 0.007
        cls_score_text = cls_score_text.softmax(dim=1)
        # ------------------------------------------------
        # using clip to inference
        if self.ensemble:
            # bboxes = rois
            # clip_image_features_ensemble = []
            # for i in range(len(img)):
            #     clip_image_features = self.img2pil2feat(img[i], bboxes[sum(num_proposals_per_img[:i]): sum(num_proposals_per_img[:i+1]), 1:])
            #     clip_image_features_single = clip_image_features.float() @ self.project_for_image_clip.T
            #     # clip_image_features_single = clip_image_features.float()
            #     clip_image_features_single = torch.nn.functional.normalize(clip_image_features_single, p=2, dim=1)
            #     clip_image_features_ensemble.append(clip_image_features_single)
            # clip_image_features_ensemble = torch.cat(clip_image_features_ensemble, dim=0)

            _, region_embeddings_image = self._bbox_forward_for_image(x, rois)
            region_embeddings_image = self.projection_for_image(region_embeddings_image)
            region_embeddings_image = torch.nn.functional.normalize(region_embeddings_image, p=2, dim=1)
            clip_image_features_ensemble = region_embeddings_image

            cls_score_image = clip_image_features_ensemble @ text_features.T
            cls_score_image = cls_score_image / 0.007
            cls_score_image[:, -1] = -1e11
            cls_score_image = cls_score_image.softmax(dim=1)
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

    # def clip_image_forward_align(self, img, bboxes, num_proposals_per_img, flag=False):
    #     cropped_images = roi_align(img, bboxes, (224, 224))
    #     image_features = self.clip_model.encode_image(cropped_images).float()
    #     image_features = image_features @ self.project_for_image_clip.T
    #     return image_features

    # get all clip finetune feature in single img (may have one more bboxes in one image)
    def get_single_clip_finetune_feature(self, img, bboxes):
        clip_image_features_single = self.img2pil2feat(img, bboxes)
        clip_image_features_single = clip_image_features_single.float()
        # alpha = 1.0
        # clip_image_features_single = clip_image_features_single @ (self.project_for_image_clip * alpha + \
        #                                                            torch.eye(self.project_for_image_clip.shape[-1],
        #                                                                      device=self.device) * (1.0 - alpha)).T

        # clip_image_features_single = self.project_lora_b(self.project_lora_a(clip_image_features_single)) * alpha + \
        #                              clip_image_features_single * (1.0 - alpha)
        # clip_image_features_single = clip_image_features_single @ self.project_for_image_clip.T
        clip_image_features_single = torch.nn.functional.normalize(clip_image_features_single, p=2, dim=1)
        return clip_image_features_single

    def forward_for_clip_finetune(self, img_no_normalize, gt_bboxes, gt_labels):
        # clip_image_features = []
        # for i in range(len(img_no_normalize)):
        #     clip_image_features_single = self.get_single_clip_finetune_feature(img_no_normalize[i], gt_bboxes[i])
        #     clip_image_features.append(clip_image_features_single)
        # clip_image_features = torch.cat(clip_image_features, dim=0)
        # labels = torch.cat(gt_labels, dim=0)
        #
        # text_features = self.text_features_for_classes
        # cls_score_text = clip_image_features @ text_features.T

        clip_image_features = []
        for i in range(len(img_no_normalize)):
            clip_image_features_single = self.img2pil2feat(img_no_normalize[i], gt_bboxes[i])
            clip_image_features.append(clip_image_features_single)
        clip_image_features = torch.cat(clip_image_features, dim=0)
        labels = torch.cat(gt_labels, dim=0)
        cls_score_text = clip_image_features
        # cls_score_text[:, self.novel_label_ids] = -1e11
        text_cls_loss = F.cross_entropy(cls_score_text / 0.01, labels,
                                        weight=torch.Tensor([1.0, 1.0, 1.0]).cuda(), reduction='mean')
        return dict({"clip_finetune_loss": text_cls_loss})

    def forward_for_clip_finetune_test(self, img_no_normalize, gt_bboxes, gt_labels):
        # clip_image_features = []
        # for i in range(len(img_no_normalize)):
        #     clip_image_features_single = self.get_single_clip_finetune_feature(img_no_normalize[i], gt_bboxes[i])
        #     clip_image_features.append(clip_image_features_single)
        # clip_image_features = torch.cat(clip_image_features, dim=0)
        # labels = torch.cat(gt_labels, dim=0)
        #
        # text_features = self.text_features_for_classes
        # cls_score_text = clip_image_features @ text_features.T

        clip_image_features = []
        for i in range(len(img_no_normalize)):
            clip_image_features_single = self.img2pil2feat(img_no_normalize[i], gt_bboxes[i])
            clip_image_features.append(clip_image_features_single)
        clip_image_features = torch.cat(clip_image_features, dim=0)
        labels = torch.cat(gt_labels, dim=0)
        cls_score_text = clip_image_features
        
        res = dict({
            "pred": cls_score_text.max(dim=-1)[1].data.cpu(),
            "gdth": labels.data.cpu(),
            # "feat": clip_image_features.data.cpu()
        })
        return [res]


    def img2pil2feat(self, img, boxs, name=None):
        img = np.array(img.detach().cpu()).astype(np.uint8)
        img = Image.fromarray(img.transpose(1, 2, 0))
        img_shape = img.size
        boxs = torch.dstack(
            [torch.floor(boxs[:, 0] - 0.001), torch.floor(boxs[:, 1] - 0.001), torch.ceil(boxs[:, 2] + 0.001),
             torch.ceil(boxs[:, 3] + 0.001)]).squeeze(0)
        # boxs = torch.dstack([torch.floor(boxs[:,0]),torch.floor(boxs[:,1]),torch.ceil(boxs[:,2]),torch.ceil(boxs[:,3])]).squeeze(0)
        boxs[:, [0, 2]].clamp_(min=0, max=img_shape[0])
        boxs[:, [1, 3]].clamp_(min=0, max=img_shape[1])
        boxs = boxs.detach().cpu().numpy()
        # print(boxs)
        preprocessed = []
        i = 0
        for box in boxs:
            try:
                croped = img.crop(box)
            except:
                print(box)
            # croped.save(name+f'_pil_{i}.jpg')
            i += 1
            croped = self.preprocess(croped)
            preprocessed.append(croped)

        preprocessed = torch.stack(preprocessed).to(self.device)
        # features = self.clip_model.encode_image(preprocessed)
        features = self.encode_image(preprocessed)
        return features

    def encode_image(self, image):
        # x = self.clip_model.encode_image(image)
        # x = self.clip_lora(image)
        x = self.resnet_grad(image)
        return x

class resnet_grad(nn.Module):
    def __init__(self, text_features_for_classes):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.project = nn.Linear(1000, 3)
        # self.project = nn.Linear(1000, 512)
        self.text_features_for_classes = text_features_for_classes

    def forward(self, x):
        x = self.resnet(x)
        x = self.project(x)
        # x = torch.nn.functional.normalize(x, p=2, dim=1)
        # text_features = self.text_features_for_classes
        # cls_score_text = x @ text_features.T
        cls_score_text = x
        return cls_score_text



