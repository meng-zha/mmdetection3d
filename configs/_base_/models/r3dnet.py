model = dict(
    type='R3DNet',
    backbone=dict(
        type='PointNet2SAMSGFP',
        in_channels=4,
        num_points=(4096, 1024, 256, 64),
        radii=((0.1,0.5),(0.5,1.0),(1.0,2.0),(2.0,4.0)),
        num_samples=((32, 64), (32, 64), (32, 64), (32, 32)),
        sa_channels=(((16, 16, 32), (32, 32, 64)),
                    ((64, 64, 128), (64, 96, 128)),
                    ((128, 196, 256), (128, 196, 256)),
                    ((256,256,512),(256,256,512))),
        aggregation_channels=(64, 128, 256, 512),
        fps_mods=(('D-FPS'),('D-FPS'),('D-FPS'),('D-FPS')),
        fp_channels = ((512,512),(512,512),(256,256),(256,256)),
        fps_sample_range_lists=((-1), (-1), (-1), (-1)),
        norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.1),
        sa_cfg=dict(
            type='PointSAModuleMSG',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=False)),
    bbox_head=dict(
        type='R3D3DHead',
        in_channels=256,
        hidden_module_cfg=dict(
            type='PointSAModuleMSG',
            num_point=4096,
            radii=(4.8, 6.4),
            sample_nums=(16, 32),
            mlp_channels=((256, 128), (256, 128)),
            norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.1),
            use_xyz=True,
            fps_mod = ('F-FPS',),
            normalize_xyz=False,
            bias=True),
        vote_module_cfg=dict(
            in_channels=256,
            num_points=256,
            gt_per_seed=1,
            conv_channels=(128, ),
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.1),
            with_res_feat=False,
            vote_xyz_range=(3.0, 3.0, 2.0)),
        vote_aggregation_cfg=dict(
            type='PointSAModuleMSG',
            num_point=256,
            radii=(4.8, 6.4),
            sample_nums=(16, 32),
            mlp_channels=((256, 256, 256, 512), (256, 256, 512, 1024)),
            norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.1),
            use_xyz=True,
            fps_mod = ('F-FPS',),
            normalize_xyz=False,
            bias=True),
        pred_layer_cfg=dict(
            in_channels=1536,
            shared_conv_channels=(512, 128),
            cls_conv_channels=(128, ),
            reg_conv_channels=(128, ),
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.1),
            bias=True),
        sem_layer_cfg=dict(
            in_channels=259,
            cls_conv_channels=(128, 64, 1),
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.1),
            act_cfg=dict(type='ReLU'),
            bias=True),
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.1),
        objectness_loss=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        sem_loss=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        center_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=1.0),
        dir_class_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        dir_res_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=1.0),
        size_res_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=1.0),
        corner_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=1.0),
        offset_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=1.0),
        vote_loss=dict(type='SmoothL1Loss', reduction='sum', loss_weight=1.0)))

# model training and testing settings
train_cfg = dict(
    sample_mod='spec', pos_distance_thr=10.0, expand_dims_length=0.05, keep_thr = 0.01)
test_cfg = dict(
    nms_cfg=dict(type='nms', iou_thr=0.1),
    sample_mod='spec',
    score_thr=0.0,
    keep_thr=0.01,
    per_class_proposal=True,
    expand_dims_length=0.05,
    with_hidden=True,
    max_output_num=100)

# optimizer
# This schedule is mainly used by models on indoor dataset,
# e.g., VoteNet on SUNRGBD and ScanNet
lr = 0.002  # max learning rate
optimizer = dict(type='AdamW', lr=lr, weight_decay=0)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(policy='step', warmup=None, step=[80, 120])
# runtime settings
total_epochs = 150
