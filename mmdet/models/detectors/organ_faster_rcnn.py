from ..builder import DETECTORS
from .two_stage import TwoStageDetector
from .faster_rcnn import FasterRCNN

@DETECTORS.register_module()
class OrganFasterRCNN(FasterRCNN):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):
        super(FasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

    def forward_train(self,
                      img,
                      img_no_normalize,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):

        losses = dict()
        if "step" in kwargs.keys() and kwargs['step'] == 0:
            clip_finetune_losses = self.roi_head.forward_for_clip_finetune(img_no_normalize, gt_bboxes, gt_labels)
            losses.update(clip_finetune_losses)
        else:
            x = self.extract_feat(img)
            # RPN forward and loss
            if self.with_rpn:
                proposal_cfg = self.train_cfg.get('rpn_proposal',
                                                  self.test_cfg.rpn)
                rpn_losses, proposal_list = self.rpn_head.forward_train(
                    x,
                    img_metas,
                    gt_bboxes,
                    gt_labels=None,
                    gt_bboxes_ignore=gt_bboxes_ignore,
                    proposal_cfg=proposal_cfg)
                losses.update(rpn_losses)
            else:
                proposal_list = proposals

            roi_losses = self.roi_head.forward_train(x, img, img_no_normalize, img_metas,
                                                     proposal_list, proposal_list,
                                                     gt_bboxes, gt_labels,
                                                     gt_bboxes_ignore, gt_masks,
                                                     **kwargs)
            losses.update(roi_losses)

        return losses


    # def forward_test(self, img, img_no_normalize, img_metas, gt_bboxes, gt_labels, **kwargs):
    #     return self.roi_head.forward_for_clip_finetune_test(img_no_normalize, gt_bboxes, gt_labels)
