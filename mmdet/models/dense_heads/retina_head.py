import torch.nn as nn
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init

from ..builder import HEADS
from .anchor_head import AnchorHead
import time
import torch
import torch.nn.functional as F

@HEADS.register_module()
class RetinaHead(AnchorHead):
    r"""An anchor-based head used in `RetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Example:
        >>> import torch
        >>> self = RetinaHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes)
        >>> assert box_per_anchor == 4
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(RetinaHead, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            **kwargs)

        # L10 add for multi
        import clip
        from mmdet.models.roi_heads.class_name import ORGAN_DESC
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * 512, # L10 change for multi
            # self.num_anchors * (self.cls_out_channels + 1),
            3,
            padding=1)
        # load text feature
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load('ViT-B/32', device)
        self.clip_model.eval()
        for child in self.clip_model.children():
            for param in child.parameters():
                param.requires_grad = False
        self.text_features_for_classes = []
        time_start = time.time()
        self.text_features_for_classes = torch.cat([self.clip_model.encode_text(clip.tokenize(desc).to(device)).detach() for desc in ORGAN_DESC], dim=0).float()
        self.text_features_for_classes = F.normalize(self.text_features_for_classes, p=2, dim=-1)
        print('retina organ head text embedding finished, {} passed'.format(time.time() - time_start))
        print(self.text_features_for_classes.shape)

        # self.bg_embedding = nn.Linear(1, 512)
        # nn.init.xavier_uniform_(self.bg_embedding.weight)
        # nn.init.constant_(self.bg_embedding.bias, 0)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        # L10 use for multimodal
        cls_feat_embed = self.retina_cls(cls_feat)
        n, c, h, w = cls_feat_embed.shape
        cls_feat_embed = cls_feat_embed.reshape(n, self.num_anchors, c // self.num_anchors, h, w)  # nbchw
        # cls_feat_embed = F.normalize(cls_feat_embed, p=2, dim=2)  # nbchw
        text_embed = F.normalize(self.text_features_for_classes, p=2, dim=1)  # tc
        cls_score = torch.einsum('nbchw, tc -> nbthw', (cls_feat_embed, text_embed))
        # cls_score = (cls_score / 0.1)
        # cls_score = (cls_score / 0.07).softmax(dim=2)
        cls_score = cls_score.reshape(n, -1, h, w)  # /  0.01

        # # original
        # # L10 use for normal
        # cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred