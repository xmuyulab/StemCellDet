_base_ = 'organ_faster_rcnn_r50_fpn_base.py'
model = dict(roi_head=dict(bbox_head=dict(ensemble=False)))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(
        type='Resize',
        img_scale=[(896, 896)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'img_no_normalize', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(896, 896),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img', 'img_no_normalize']),
            dict(type='Collect', keys=['img', 'img_no_normalize']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(dataset=dict(ann_file='./data/test.csv',
                            pipeline=train_pipeline)),
    val=dict(ann_file='./data/test.csv',
             pipeline=test_pipeline),
    test=dict(ann_file='./data/test.csv',
              pipeline=test_pipeline)
)

test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=16,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    # rcnn=dict(
    #     score_thr=0.05,
    #     nms=dict(type='nms', iou_threshold=0.5),
    #     max_per_img=100)
)

# load_from = 'data/epoch_45.pth'
# load_from = 'data/epoch_6.pth'
# load_from='workdirs/vild_ens_20e_fg_bg_5_10_end_vild_tmp/epoch_11.pth'
# load_from = 'workdirs/vild_ens_20e_fg_bg_5_10_end_vild_triplet_11_0.5m_0.8w/epoch_16.pth'
# load_from = 'workdirs/vild_ens_20e_fg_bg_5_10_end_vild_grl_forwardmove_0.8w/epoch_15.pth'

load_from = 'workdirs/vild_ens_20e_fg_bg_5_10_end_tmp_tmp/epoch_3_432.pth'
# load_from = '/home/lih/work/projects/organoid_ovod/workdirs/detpro_final_exp_bugfix/vild_ens_20e_fg_bg_5_10_end_base_alldata/epoch_10.pth'
# load_from = '/home/lih/work/projects/organoid_ovod/workdirs/detpro_final_exp_bugfix/vild_ens_20e_fg_bg_5_10_end_vild_vild_0.005/epoch_7.pth'
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
