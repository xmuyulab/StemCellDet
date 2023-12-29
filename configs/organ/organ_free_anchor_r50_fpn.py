_base_ = 'organ_retinanet_r50_fpn.py'
model = dict(
    neck=dict(
        start_level=0,
    ),
    bbox_head=dict(
        _delete_=True,
        type='FreeAnchorRetinaHead',
        num_classes=3,  # 3 will report error
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=8,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        # anchor_generator=dict(
        #     type='AnchorGenerator',
        #     octave_base_scale=4,
        #     scales_per_octave=3,
        #     ratios=[0.5, 1.0, 2.0],
        #     strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=0.75)))

optim_wrapper = dict(clip_grad=dict(max_norm=35, norm_type=2))

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
)
checkpoint_config = dict(interval=1, create_symlink=False)
evaluation = dict(interval=1, metric=['bbox'])
# load_from = 'data/epoch_45.pth'
# load_from = '/home/lih/work/projects/organoid_ovod/workdirs/detpro_final_exp_bugfix/vild_ens_20e_fg_bg_5_10_end_vild_vild_0.005/epoch_7.pth'
# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=2.5e-05)
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=2.5e-05)
lr_config = dict(step=[8, 16])
total_epochs = 25
