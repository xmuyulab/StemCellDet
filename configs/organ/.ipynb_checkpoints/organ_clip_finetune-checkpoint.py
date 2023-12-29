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
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(
        type='Resize',
        img_scale=[(896, 896)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'img_no_normalize', 'gt_bboxes', 'gt_labels']),
]
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=1,
    train=dict(dataset=dict(pipeline=train_pipeline)),
    val=dict(ann_file='./data/test.csv',
             pipeline=test_pipeline,),
    test=dict(pipeline=test_pipeline)
)
# optimizer = dict(type='Adam', lr=1e-4)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
log_config = dict(interval=1)
lr_config = dict(
    warmup_iters=10.0,
    step=[40])
total_epochs = 50