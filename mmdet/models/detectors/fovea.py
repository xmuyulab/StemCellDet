from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class FOVEA(SingleStageDetector):
    """Implementation of `FoveaBox <https://arxiv.org/abs/1904.03797>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(FOVEA, self).__init__(backbone, neck, bbox_head, train_cfg,
                                    test_cfg, pretrained)

    # add by L10
    # add param img_no_normalize and delete, reset to the former
    def forward_train(self,
                      img,
                      img_no_normalize,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        data = super(FOVEA, self).forward_train(img, img_metas, gt_bboxes,
                                                    gt_labels, gt_bboxes_ignore)
        return data

    # add by L10
    # add param img_no_normalize and delete, reset to the former
    def simple_test(self, img, img_no_normalize, img_metas, rescale=False):
        data = super(FOVEA, self).simple_test(img, img_metas, rescale)
        return data
