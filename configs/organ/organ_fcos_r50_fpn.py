_base_ = [
    '../_base_/models/fcos_r50_fpn_gn.py',
    '../_base_/datasets/lvis_v1_instance.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]
dataset_type = 'OrganDataset'
data_root = './data/'

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
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        _delete_=True,
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'train.csv',
            img_prefix=data_root + 'img_clip_all',
            pipeline=train_pipeline
        )
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'test.csv',
        img_prefix=data_root + 'img_clip_all',
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'test.csv',
        img_prefix=data_root + 'img_clip_all',
        pipeline=test_pipeline
    )
)

train_cfg = dict(
    sampler=dict(
        type='RandomSampler',
        num=256,
        pos_fraction=0.5,
        neg_pos_ub=-1,
        add_gt_as_proposals=False
    )
)
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=16,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
)
model = dict(
    bbox_head=dict(
        num_classes=3,
    )
)

checkpoint_config = dict(interval=1, create_symlink=False)
evaluation = dict(interval=1, metric=['bbox'])
load_from = 'data/epoch_45.pth'
# load_from = 'workdirs/vild_ens_20e_fg_bg_5_10_end_vild_vild/epoch_1.pth'
# load_from = '/home/lih/work/projects/organoid_ovod/workdirs/detpro_final_exp_bugfix/vild_ens_20e_fg_bg_5_10_end_vild_vild_0.005/epoch_7.pth'
# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=2.5e-05)
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=2.5e-05)
lr_config = dict(step=[8, 16])
total_epochs = 25