point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
voxel_size = [0.4, 0.4, 0.4]
num_classes = 18
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
class_names = [
    'car',
    'truck',
    'construction_vehicle',
    'bus',
    'trailer',
    'barrier',
    'motorcycle',
    'bicycle',
    'pedestrian',
    'traffic_cone',
]

mapping_table = {
    0: 1, # barrier
    1: 1, # traffic barrier
    2: 2, # bicycle
    3: 3, # bus
    4: 4, # car
    5: 4, # vehicle
    6: 4, # sedan
    7: 4, # SUV
    8: 5, # construction vehicle
    9: 5, # crane
    10: 6, # motorcycle
    11: 7, # pedestrian
    12: 7, # person
    13: 8, # traffic cone
    14: 9, # trailer
    15: 9, # delivery trailer
    16: 10, # truck
    17: 11, # driveable surface
    18: 11, # road
    19: 12, # water
    20: 12, # river
    21: 12, # lake
    22: 13, # sidewalk
    23: 14, # terrain
    24: 14, # grass
    25: 15, # building
    26: 15, # wall,
    27: 15, # traffic light
    28: 15, # sign
    29: 15, # parking meter
    30: 15, # hydrant
    31: 15, # fence
    32: 16, # vegetation
    33: 16, # tree
    34: 17, # sky
    255: 17, # free
}

CLASS_NAMES = [
    'others',
    'barrier',
    'bicycle',
    'bus',
    'car',
    'construction_vehicle',
    'motorcycle',
    'pedestrian',
    'traffic_cone',
    'trailer',
    'truck',
    'driveable_surface',
    'other_flat',
    'sidewalk',
    'terrain',
    'manmade',
    'vegetation',
    'free',
]

occupancy_classes= ['others','barrier','bicycle','bus','car',
                    'construction_vehicle','motorcycle','pedestrian',
                    'traffic_cone', 'trailer', 'truck',
                    'driveable_surface', 'other_flat', 'sidewalk',
                    'terrain', 'manmade', 'vegetation','free']

class_weight_multiclass = [
    1.552648813025149,
    1.477680635715412,
    1.789915946148316,
    1.454376653104962,
    1.283242744137921,
    1.583160056748120,
    1.758171915228669,
    1.468604241657418,
    1.651769160217543,
    1.454675968105020,
    1.369895420004945,
    1.125140370991227,
    1.399044660772846,
    1.203105344914611,
    1.191157881795851,
    1.155987296237377,
    1.150134564832974,
    1.000000000000000,
]



input_modality = dict(
    use_lidar=False,
    use_camera=True, 
    use_radar=False, 
    use_map=False, 
    use_external=True
)

_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200
pillar_h = 16
channels = 16
queue_length = 1
use_padding = False
use_temporal = None
scales = None
use_camera_mask = True
use_lidar_mask = False
use_refine_feat_loss = False
refine_feat_loss_weight = 10

use_temporal_self_attention = False
if use_temporal_self_attention:
    attn_cfgs = [
        dict(type='TemporalSelfAttention', embed_dims=_dim_, num_levels=1),
        dict(
            type='SpatialCrossAttention',
            pc_range=point_cloud_range,
            deformable_attention=dict(
                type='MSDeformableAttention3D',
                embed_dims=_dim_,
                num_points=8,
                num_levels=_num_levels_,
            ),
            embed_dims=_dim_,
        ),
    ]
    operation_order = ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
else:
    attn_cfgs = [
        dict(
            type='SpatialCrossAttention',
            pc_range=point_cloud_range,
            deformable_attention=dict(
                type='MSDeformableAttention3D',
                embed_dims=_dim_,
                num_points=8,
                num_levels=_num_levels_,
            ),
            embed_dims=_dim_,
        )
    ]
    operation_order = ('cross_attn', 'norm', 'ffn', 'norm')

data_root = 'data/occ3d/'
text_embedding_root = 'data/occ3d/text_embedding/'
text_embedding_file = 'query_128.json'
language_gt_root = 'cat_seg_gts_qwen_scene'
img_scales = [1.0]
dataset_type = 'NuSceneOcc'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeImages', img_size=(256, 704)),
    dict(type='LoadOccGTFromFileNuScenes', data_root=data_root),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(
        type='LoadOVOGTFromFile',
        ovo_gt_root=language_gt_root,
        data_root=data_root,
        scene_specific=True,
        text_embedding_file='vocab_128.json',
        ignore_label=34
    ),
    dict(type='RandomScaleImageMultiViewImage', scales=img_scales),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
    ),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='CustomCollect3D',
        keys=[
            'img',
            'voxel_semantics',
            'mask_lidar',
            'mask_camera',
            'ovo_gt',
            'ovo_gt_mask'
        ],
        meta_keys=(
            'filename',
            'pts_filename',
            'occ_gt_path',
            'scene_token',
            'frame_idx',
            'scene_idx',
            'sample_idx',
            'ori_shape',
            'img_shape',
            'pad_shape',
            'lidar2img',
            'ego2lidar',
            'ego2global',
            'cam_intrinsic',
            'lidar2cam',
            'cam2img',
            'can_bus',
            'post_rot',
            'post_tran',
            'text_embedding_file'
        ),
    ),
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeImages', img_size=(256, 704)),
    dict(type='LoadOccGTFromFileNuScenes', data_root=data_root),
    dict(type='RandomScaleImageMultiViewImage', scales=img_scales),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='CustomCollect3D',
        keys=[
            'img',
            'voxel_semantics',
            'mask_lidar',
            'mask_camera',
        ],
        meta_keys=(
            'filename',
            'pts_filename',
            'occ_gt_path',
            'scene_token',
            'frame_idx',
            'scene_idx',
            'sample_idx',
            'ori_shape',
            'img_shape',
            'pad_shape',
            'lidar2img',
            'ego2lidar',
            'ego2global',
            'cam_intrinsic',
            'lidar2cam',
            'cam2img',
            'can_bus',
            'post_rot',
            'post_tran'
        ),
    ),
]

data = dict(
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'occ_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        box_type_3d='LiDAR',
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'occ_infos_temporal_val.pkl',
        pipeline=test_pipeline,
        bev_size=(bev_h_, bev_w_),
        classes=class_names,
        modality=input_modality,
        # below are evaluation settings
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        CLASS_NAMES=CLASS_NAMES,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'occ_infos_temporal_val.pkl',
        pipeline=test_pipeline,
        bev_size=(bev_h_, bev_w_),
        classes=class_names,
        modality=input_modality,
        # below are evaluation settings
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        CLASS_NAMES=CLASS_NAMES,
    ),
)

batch_size=1
train_dataloader_config = dict(
    batch_size=batch_size,
    num_workers=4)

test_dataloader_config = dict(
    batch_size=batch_size,
    num_workers=4)

model = dict(
    type='BEVFormerOVO',
    use_grid_mask=True,
    video_test_mode=True,
    queue_length=queue_length,
    save_results=False,
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(
            type='DCNv2', deform_groups=1, fallback_on_stride=False
        ),  # original DCNv2 will print log when perform load_state_dict
        stage_with_dcn=(False, False, True, True),
    ),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True,
    ),
    img_view_transformer=dict(
        type='BEVOccHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_classes=num_classes,
        transformer=dict(
            type='BEVOccTransformer',
            pillar_h=pillar_h,
            num_classes=num_classes,
            bev_h=bev_h_,
            bev_w=bev_w_,
            channels=channels,
            pc_range=point_cloud_range,
            voxel_size=voxel_size,
            norm_cfg=dict(type='BN',),
            norm_cfg_3d=dict(type='BN2d',),
            rotate_prev_bev=False,
            use_shift=False,
            use_can_bus=False,
            embed_dims=_dim_,
            queue_length=queue_length,
            use_padding=use_padding,
            use_temporal=use_temporal,
            scales=scales,
            encoder=dict(
                type='BEVFormerEncoder',            
                bev_h=bev_h_,
                bev_w=bev_w_,
                num_layers=6,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                resize=True,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    bev_h=bev_h_,
                    bev_w=bev_w_,
                    attn_cfgs=attn_cfgs,
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=operation_order,
                ),
            ),
        ),
    ),
    occ_head=dict(
        type='OVOHead',
        embed_dims=_dim_,
        mid_channels=64,
        mid_channels_occ=32,
        mid_channels_language=64,
        language_channels=128,
        num_classes=18,
        scene_specific=True,
        text_embedding=text_embedding_root + text_embedding_file,
        mapping_table=mapping_table,
        loss_occ=dict(
            type='CrossEntropyLoss',
            # class_weight=class_weight_multiclass,
            use_sigmoid=False,
            loss_weight=1.0,
        ),
    )
)

"""Training params."""
learning_rate=3e-4
training_steps=93000

optimizer = dict(
    type="AdamW",
    lr=learning_rate,
    weight_decay=0.01
)

lr_scheduler = dict(
    type="OneCycleLR",
    max_lr=learning_rate,
    total_steps=training_steps + 10,
    pct_start=0.05,
    cycle_momentum=False,
    anneal_strategy="cos",
    interval="step",
    frequency=1
)

load_from = 'ckpts/r101_dcn_fcos3d_pretrain.pth'