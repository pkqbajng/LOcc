# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
    'input_size': (512, 1408),
    'src_size': (900, 1600),
    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

occupancy_classes= ['others','barrier','bicycle','bus','car',
                    'construction_vehicle','motorcycle','pedestrian',
                    'traffic_cone', 'trailer', 'truck',
                    'driveable_surface', 'other_flat', 'sidewalk',
                    'terrain', 'manmade', 'vegetation','free']

grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 0.4],
    'depth': [1.0, 45.0, 0.5],
}

# Data
dataset_type = 'NuScenesDatasetOccpancy'
data_root = 'data/occ3d/'
text_embedding_root = 'data/occ3d/text_embedding/'
text_embedding_file = 'query_128.json'
language_gt_root = 'san_gts_qwen_scene'
file_client_args = dict(backend='disk')

img_info_prototype = 'bevdet'
sequential = False
multi_adj_frame_id_cfg = (1, 1+1, 1)
batch_size=1

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=sequential),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=True),
    dict(
        type='LoadOVOGTFromFile',
        ovo_gt_root=language_gt_root,
        scene_specific=True,
        text_embedding_file='vocab_128.json',
        ignore_label=34
    ),
    dict(type='LoadOccGTFromFile'),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_depth', 'voxel_semantics',
                                'mask_lidar', 'mask_camera', 'ovo_gt', 'ovo_gt_mask'],
        meta_keys=['text_embedding_file'])
]

test_pipeline = [
    dict(
        type='PrepareImageInputs', 
        data_config=data_config, 
        sequential=sequential),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(type='LoadOccGTFromFile'),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_depth', 'voxel_semantics',
                                'mask_lidar', 'mask_camera'])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    data_root=data_root,
    classes=class_names,
    modality=input_modality,
    stereo=True,
    filter_empty_gt=False,
    img_info_prototype=img_info_prototype,
)

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=data_root + 'bevdetv2-nuscenes_infos_val.pkl')

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'bevdetv2-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
       # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'train', 'test']:
    data[key].update(share_data_config)

train_dataloader_config = dict(
    batch_size=batch_size,
    num_workers=4)

test_dataloader_config = dict(
    batch_size=batch_size,
    num_workers=4)


# model

numC_Trans = 64
use_mask = True
num_classes=18

mid_channels_occ=32
mid_channels_language=128
language_channels=128


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

model = dict(
    type='BEVDetOVO',
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='./ckpts/resnet50-0676ba61.pth')
    ),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    depth_net=dict(
        type='DepthNet',
        in_channels=256,
        context_channels=numC_Trans,
        grid_config=grid_config,
        loss_depth_weight=0.05,
        downsample=8,
    ),
    img_view_transformer=dict(
        type='LSSViewTransformer',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        downsample=8,
        out_channels=numC_Trans,
        accelerate=False,
        sid=False,
        collapse_z=False,
    ),
    img_bev_encoder_backbone=dict(
        type='CustomResNet3D',
        numC_input=numC_Trans,
        num_layer=[1, 2, 4],
        with_cp=False,
        num_channels=[numC_Trans, numC_Trans*2, numC_Trans*4],
        stride=[1, 2, 2],
        backbone_output_ids=[0, 1, 2]),
    img_bev_encoder_neck=dict(type='LSSFPN3D',
                              in_channels=numC_Trans*7,
                              out_channels=numC_Trans),
    occ_head=dict(
        type='OVOHead',
        in_dim=numC_Trans,
        mid_channels_occ=mid_channels_occ,
        mid_channels_language=mid_channels_language,
        language_channels=language_channels,
        text_embedding=text_embedding_root + text_embedding_file,
        num_classes=18,
        mapping_table=mapping_table,
        use_mask=True,
        scene_specific=True
    )
)

"""Training params."""
learning_rate=3e-4
training_steps=93000
pretrain=False

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

load_from="ckpts/bevstereo-san-1408x512.pth"