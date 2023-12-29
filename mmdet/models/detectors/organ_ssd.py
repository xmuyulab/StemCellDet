from .single_stage import SingleStageDetector
from ..builder import DETECTORS

@DETECTORS.register_module()
class OrganSSD(SingleStageDetector):
    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(OrganSSD, self).__init__(backbone=backbone,
                                       neck=neck,
                                       bbox_head=bbox_head,
                                       train_cfg=train_cfg,
                                       test_cfg=test_cfg,
                                       pretrained=pretrained)
    # add by L10
    # add param img_no_normalize and delete, reset to the former
    def forward_train(self,
                      img,
                      img_no_normalize,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        data = super(OrganSSD, self).forward_train(img, img_metas, gt_bboxes,
                                                   gt_labels, gt_bboxes_ignore)
        return data

    # add by L10
    # add param img_no_normalize and delete, reset to the former
    def simple_test(self, img, img_no_normalize, img_metas, rescale=False):
        data = super(OrganSSD, self).simple_test(img, img_metas, rescale)
        return data
