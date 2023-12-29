from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class RetinaNet(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaNet, self).__init__(backbone, neck, bbox_head, train_cfg,
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
        data = super(RetinaNet, self).forward_train(img, img_metas, gt_bboxes,
                                                    gt_labels, gt_bboxes_ignore)
        return data

    # add by L10
    # add param img_no_normalize and delete, reset to the former
    def simple_test(self, img, img_no_normalize, img_metas, rescale=False):
        data = super(RetinaNet, self).simple_test(img, img_metas, rescale)
        return data
