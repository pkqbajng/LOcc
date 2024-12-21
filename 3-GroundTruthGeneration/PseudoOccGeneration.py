import os
import sys
import pdb
import time
import yaml
import torch
import mmcv
import numba
import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from tqdm import tqdm
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from mmcv.ops.points_in_boxes import (points_in_boxes_all, points_in_boxes_cpu,
                                      points_in_boxes_part)
from scipy.spatial.transform import Rotation
from collections import Counter
import pickle
import open3d
import open3d as o3d
from copy import deepcopy

camera_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_FRONT_RIGHT']

def rt2mat(translation, quaternion=None, inverse=False, rotation=None):
    R = Quaternion(quaternion).rotation_matrix if rotation is None else rotation
    T = np.array(translation)
    if inverse:
        R = R.T
        T = -R @ T
    mat = np.eye(4)
    mat[:3, :3] = R
    mat[:3, 3] = T
    return mat

def test_mask_order(occ_size, occ_index, occ_mask):
    gt_ = np.zeros(occ_size)
    x = np.linspace(0, gt_.shape[0] - 1, gt_.shape[0])
    y = np.linspace(0, gt_.shape[1] - 1, gt_.shape[1])
    z = np.linspace(0, gt_.shape[2] - 1, gt_.shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    vv = np.stack([X, Y, Z], axis=-1)
    print(occ_index.shape)
    print(vv.shape)

    vv = vv.reshape([-1, 3])
    occ_mask = occ_mask.reshape([-1, ]).astype(bool)
    print(vv.shape)
    print(vv[occ_mask].shape)
    print(vv[occ_mask][:5, :])
    print(occ_index[:5, :])
    print(np.array_equal(vv[occ_mask], occ_index))

# input: (N, 4), output: (4, N)
def lidar_to_world_to_lidar(pc,lidar_calibrated_sensor,lidar_ego_pose,
    cam_calibrated_sensor,
    cam_ego_pose):

    pc = LidarPointCloud(pc.T)
    pc.rotate(Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_calibrated_sensor['translation']))

    pc.rotate(Quaternion(lidar_ego_pose['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_ego_pose['translation']))

    pc.translate(-np.array(cam_ego_pose['translation']))
    pc.rotate(Quaternion(cam_ego_pose['rotation']).rotation_matrix.T)

    pc.translate(-np.array(cam_calibrated_sensor['translation']))
    pc.rotate(Quaternion(cam_calibrated_sensor['rotation']).rotation_matrix.T)

    return pc

def lidar_to_world_to_ego(pc,lidar_calibrated_sensor,lidar_ego_pose,
    cam_calibrated_sensor,
    cam_ego_pose):

    pc = LidarPointCloud(pc.T)
    pc.rotate(Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_calibrated_sensor['translation']))

    pc.rotate(Quaternion(lidar_ego_pose['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_ego_pose['translation']))

    pc.translate(-np.array(cam_ego_pose['translation']))
    pc.rotate(Quaternion(cam_ego_pose['rotation']).rotation_matrix.T)

    return pc

def lidar2vehicle(pc,lidar_calibrated_sensor):
    pc = LidarPointCloud(pc.T)
    pc.rotate(Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_calibrated_sensor['translation']))
    return pc
    
def get_occ_info(index):
    inputs = {}
    rec = nusc.get('sample', index)
    scene_name = nusc.get('scene', rec['scene_token'])['name']
    print(scene_name)

    proj_matrix = []
    filenames = []
    for cam_name in camera_names:
        cam_sample = nusc.get('sample_data', rec['data'][cam_name])
        filename = cam_sample['filename']
        filenames.append(filename)

        ego_spatial = nusc.get('calibrated_sensor', cam_sample['calibrated_sensor_token'])
        cam2ego = rt2mat(ego_spatial['translation'], ego_spatial['rotation']).astype(np.float32)
        ego2cam = rt2mat(ego_spatial['translation'], ego_spatial['rotation'], inverse=True).astype(np.float32)

        K = np.eye(4).astype(np.float32)
        K[:3, :3] = ego_spatial['camera_intrinsic']

        T = np.dot(K, ego2cam)
        proj_matrix.append(T)
    inputs['proj_matrix'] = proj_matrix
    return inputs, filenames, scene_name

def get_san_seg_result(proj_img_index, proj_uv, filenames, args):
    seg_results = []
    for filename in filenames:
        seg_path = os.path.join(args.seg_root, filename.replace('.jpg', '.png'))
        
        seg_result = cv2.imread(seg_path, -1)
        seg_result = seg_result.astype(np.uint8)
        seg_results.append(seg_result)
        
    u, v = proj_uv[:, 0], proj_uv[:, 1]
    seg_return = np.zeros_like(proj_img_index) + 255
    for i in range(len(seg_results)):
        index_mask = proj_img_index == i
        u_mask, v_mask = u[index_mask], v[index_mask]
        seg_return[index_mask] = seg_results[i][(v_mask).astype(np.int16), (u_mask).astype(np.int16)]
    
    return seg_return

def vehicle2img(coords, proj_matrix):
    '''
    input coords: (N, 3), proj_matrix: list
    output proj_info: (N, 3), (proj_img_index, u, v)
           proj_mask: (N, )
           u, v: float32, for feature interpolate
    '''
    proj_info = np.zeros_like(coords) - 1
    ones_column = np.ones((coords.shape[0], 1))
    coords = np.concatenate((coords, ones_column), axis=-1)
    center = (800, 450)

    for i in range(len(proj_matrix)):
        proj = proj_matrix[i]
        img_coords = np.dot(proj, coords.T)
        uv_coords = img_coords[:2, :] / img_coords[2, :]
        uv_coords = uv_coords.T
        depth_coords = img_coords[2, :].T

        valid = (uv_coords[:, 0] >= 0) & (uv_coords[:, 0] < 1600) & (uv_coords[:, 1] >= 0) & (uv_coords[:, 1] < 900) & (depth_coords > 0)
        valid_indices = np.where(valid)[0]
        for valid_indice in valid_indices:
            if proj_info[valid_indice, 0] < 0:
                proj_info[valid_indice, 0] = i
                proj_info[valid_indice, 1:] = uv_coords[valid_indice, :]
            else:
                old = (proj_info[valid_indice, 1] - center[0]) ** 2 + (proj_info[valid_indice, 2] - center[1]) ** 2
                new = (uv_coords[valid_indice, 0] - center[0]) ** 2 + (uv_coords[valid_indice, 1] - center[1]) ** 2
                if new < old:
                    proj_info[valid_indice, 0] = i
                    proj_info[valid_indice, 1:] = uv_coords[valid_indice, :]
    
    proj_mask = (proj_info[:, 0] < 0) | (proj_info[:, 1] < 0) | (proj_info[:, 2] < 0)
    proj_mask = ~proj_mask
    return proj_info, proj_mask

def main(nusc, val_list, indice, nuscenesyaml, args, config):

    save_path = args.save_path
    data_root = args.data_root
    learning_map = nuscenesyaml['learning_map']
    voxel_size = config['voxel_size']
    pc_range = config['pc_range']
    occ_size = config['occ_size']

    my_scene = nusc.scene[indice]
    sensor = 'LIDAR_TOP'

    if args.split == 'train':
        if my_scene['token'] in val_list:
            return
    elif args.split == 'val':
        if my_scene['token'] not in val_list:
            return
    elif args.split == 'all':
        pass
    else:
        raise NotImplementedError


    # load the first sample to start
    first_sample_token = my_scene['first_sample_token']
    my_sample = nusc.get('sample', first_sample_token)
    lidar_data = nusc.get('sample_data', my_sample['data'][sensor])
    # ego_pose: ego to global; calibrated_sensor: lidar to ego
    lidar_ego_pose0 = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    lidar_calibrated_sensor0 = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])

    # collect LiDAR sequence
    dict_list = []

    frame = 0
    while True:
        if lidar_data['is_key_frame']:
            ############################# get boxes ##########################
            lidar_path, boxes, _ = nusc.get_sample_data(lidar_data['token'])
            boxes_token = [box.token for box in boxes]
            object_tokens = [nusc.get('sample_annotation', box_token)['instance_token'] for box_token in boxes_token]
            object_category = [nusc.get('sample_annotation', box_token)['category_name'] for box_token in boxes_token]

            ############################# get object categories ##########################
            converted_object_category = []
            for category in object_category:
                for (j, label) in enumerate(nuscenesyaml['labels']):
                    if category == nuscenesyaml['labels'][label]:
                        converted_object_category.append(np.vectorize(learning_map.__getitem__)(label).item())

            ############################# get bbox attributes ##########################
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0]
                            for b in boxes]).reshape(-1, 1)
            gt_bbox_3d = np.concatenate([locs, dims, rots], axis=1).astype(np.float32)
            gt_bbox_3d[:, 6] += np.pi / 2.
            gt_bbox_3d[:, 2] -= dims[:, 2] / 2.
            gt_bbox_3d[:, 2] = gt_bbox_3d[:, 2] - 0.1  # Move the bbox slightly down in the z direction
            gt_bbox_3d[:, 3:6] = gt_bbox_3d[:, 3:6] * 1.1 # Slightly expand the bbox to wrap all object points

            ############################# get LiDAR points with semantics ##########################
            pc_file_name = lidar_data['filename'] # load LiDAR names
            pc0 = np.fromfile(os.path.join(data_root, pc_file_name),
                            dtype=np.float32,
                            count=-1).reshape(-1, 5)[..., :4]
            
            lidar_sd_token = lidar_data['token']
            lidarseg_labels_filename = os.path.join(nusc.dataroot,
                                                    nusc.get('lidarseg', lidar_sd_token)['filename'])

            points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
            points_label = np.vectorize(learning_map.__getitem__)(points_label)

            pc_with_semantic = np.concatenate([pc0[:, :3], points_label], axis=1)

            ############################# cut out movable object points and masks ##########################
            points_in_boxes = points_in_boxes_cpu(torch.from_numpy(pc0[:, :3][np.newaxis, :, :]),
                                                torch.from_numpy(gt_bbox_3d[np.newaxis, :]))
            # object_points_list = []
            # j = 0
            # while j < points_in_boxes.shape[-1]:
            #     object_points_mask = points_in_boxes[0][:,j].bool()
            #     object_points = pc0[object_points_mask]
            #     object_points_list.append(object_points)
            #     j = j + 1

            moving_mask = torch.ones_like(points_in_boxes)
            each_box_masks = points_in_boxes[0]
            points_in_boxes = torch.sum(points_in_boxes * moving_mask, dim=-1).bool()
            points_mask = ~(points_in_boxes[0])

            ############################# get point mask of the vehicle itself ##########################
            range = config['self_range']
            oneself_mask = torch.from_numpy((np.abs(pc0[:, 0]) > range[0]) |
                                            (np.abs(pc0[:, 1]) > range[1]) |
                                            (np.abs(pc0[:, 2]) > range[2]))

            ################################# get lidar info and mask ###################################
            lidar_ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
            lidar_calibrated_sensor = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
            
            # add occ gt unfreespace mask
            # 1. convert points in lidar coords to vehicle coords
            vehicle_coords = lidar2vehicle(pc_with_semantic.copy(), lidar_calibrated_sensor)
            tmp = vehicle_coords.points.T.copy()
            vehicle_coords = vehicle_coords.points.T[:, :3]

            # 2. range mask, reserve points in range
            range_mask = (np.abs(vehicle_coords[:, 0]) < 40.0) & (np.abs(vehicle_coords[:, 1]) < 40.0) \
               & (vehicle_coords[:, 2] > -1.0) & (vehicle_coords[:, 2] < 5.4)

            # 3. vehicle coords to voxel coords
            voxel_coords = vehicle_coords.copy()
            voxel_coords[:, 0] = (voxel_coords[:, 0] - pc_range[0]) / voxel_size
            voxel_coords[:, 1] = (voxel_coords[:, 1] - pc_range[1]) / voxel_size
            voxel_coords[:, 2] = (voxel_coords[:, 2] - pc_range[2]) / voxel_size
            voxel_coords = np.floor(voxel_coords).astype(np.int32)

            # 4. get disocclusion mask
            index = lidar_data['sample_token']
            inputs, filenames, scene_name = get_occ_info(index)

            # 5. get lidar info: vehicle_distance, proj_image_name, (u,v)
            vehicle_distance = np.sqrt(vehicle_coords[:, 0] ** 2 + vehicle_coords[:, 1] ** 2 + vehicle_coords[:, 2] ** 2)
            proj_matrix, proj_mask = vehicle2img(vehicle_coords, inputs['proj_matrix'])
            proj_mask = torch.from_numpy(proj_mask)
            proj_uv = proj_matrix[:, 1:]
            proj_img_index = proj_matrix[:, 0].astype(np.int16)

            # get seg results of projected points
            semantic_category = get_san_seg_result(proj_img_index, proj_uv, filenames, args)

            # 6. final mask  
            points_mask = points_mask & oneself_mask & proj_mask

            ################### record static scene semantic information into the dict ########################
            orin_points = pc_with_semantic.copy()
            pc_with_semantic = pc_with_semantic[points_mask]
            lidar_pc_with_semantic = lidar_to_world_to_lidar(pc_with_semantic.copy(),
                                                            lidar_calibrated_sensor.copy(),
                                                            lidar_ego_pose.copy(),
                                                            lidar_calibrated_sensor0,
                                                            lidar_ego_pose0)
            # vehicle_distance_static = vehicle_distance[points_mask]
            # proj_uv_static = proj_uv[points_mask]
            # proj_img_index_static = proj_img_index[points_mask]
            # proj_img_name_static = np.array([filenames[i] for i in proj_img_index_static])
            # # lidar_info: (distance, u, v, img_name)
            # lidar_info_static = np.concatenate([vehicle_distance_static.reshape([-1,1]), proj_uv_static], axis=-1).astype(np.float16)
            # lidar_info_static = np.concatenate([lidar_info_static, proj_img_name_static.reshape([-1,1])], axis=-1)
            lidar_info_static = semantic_category[points_mask]

            ################### record moving objects semantic information into the dict ########################
            object_points_list = []
            object_lidar_info_list = []
            j = 0
            while j < each_box_masks.shape[-1]:
                each_box_mask = each_box_masks[:, j].bool()
                box_mask = each_box_mask & oneself_mask & proj_mask

                object_points = orin_points[box_mask]
                object_points_list.append(object_points)

                # vehicle_distance_object = vehicle_distance[box_mask]
                # proj_uv_object = proj_uv[box_mask]
                # proj_img_index_object = proj_img_index[box_mask]
                # proj_img_name_object = np.array([filenames[i] for i in proj_img_index_object])
                # # lidar_info: (distance, u, v, img_name)
                # lidar_info_object = np.concatenate([vehicle_distance_object.reshape([-1,1]), proj_uv_object], axis=-1).astype(np.float16)
                # lidar_info_object = np.concatenate([lidar_info_object, proj_img_name_object.reshape([-1,1])], axis=-1)
                lidar_info_object = semantic_category[box_mask]
                object_lidar_info_list.append(lidar_info_object)
                j = j + 1

            ################## record key frame information into a dict  #########################
            dict = {"object_tokens": object_tokens,
                    "object_points_list": object_points_list,
                    "object_lidar_info_list": object_lidar_info_list,
                    "lidar_pc_with_semantic": lidar_pc_with_semantic.points,
                    "lidar_ego_pose": lidar_ego_pose,
                    "lidar_calibrated_sensor": lidar_calibrated_sensor,
                    "lidar_token": lidar_data['token'],
                    "is_key_frame": lidar_data['is_key_frame'],
                    "gt_bbox_3d": gt_bbox_3d,
                    "converted_object_category": converted_object_category,
                    "pc_file_name": pc_file_name.split('/')[-1],
                    "lidar_info_static": lidar_info_static,
                    "occ_token": index,
                    "scene_name": scene_name,
                    "proj_matrix": inputs['proj_matrix']}
            dict_list.append(dict)

            frame += 1

            ################## go to next frame of the sequence  ########################
            next_token = lidar_data['next']
            if next_token != '':
                lidar_data = nusc.get('sample_data', next_token)
            else:
                break
        else:
            ################## go to next frame of the sequence  ########################
            next_token = lidar_data['next']
            if next_token != '':
                lidar_data = nusc.get('sample_data', next_token)
            else:
                break

    ################## concatenate all semantic scene segments (only key frames)  ########################
    lidar_pc_with_semantic_list = []
    lidar_info_static_list = []
    for dict in dict_list:
        if dict['is_key_frame']:
            lidar_pc_with_semantic_list.append(dict['lidar_pc_with_semantic'])
            lidar_info_static_list.append(dict['lidar_info_static'])
    lidar_pc_with_semantic = np.concatenate(lidar_pc_with_semantic_list, axis=1).T
    lidar_info_static = np.concatenate(lidar_info_static_list, axis=0)

    ################################ concatenate all object segments   #####################################
    object_token_zoo = []
    object_semantic = []
    for dict in dict_list:
        for i,object_token in enumerate(dict['object_tokens']):
            if object_token not in object_token_zoo:
                if (dict['object_points_list'][i].shape[0] > 0):
                    object_token_zoo.append(object_token)
                    object_semantic.append(dict['converted_object_category'][i])
                else:
                    continue

    object_points_dict = {}
    object_lidar_info_dict = {}

    for query_object_token in object_token_zoo:
        object_points_dict[query_object_token] = []
        object_lidar_info_dict[query_object_token] = []
        for dict in dict_list:
            for i, object_token in enumerate(dict['object_tokens']):
                if query_object_token == object_token:
                    object_points = dict['object_points_list'][i]
                    object_lidar_info = dict['object_lidar_info_list'][i]
                    if object_points.shape[0] > 0:
                        object_points = object_points[:,:3] - dict['gt_bbox_3d'][i][:3]
                        rots = dict['gt_bbox_3d'][i][6]
                        Rot = Rotation.from_euler('z', -rots, degrees=False)
                        rotated_object_points = Rot.apply(object_points)
                        object_points_dict[query_object_token].append(rotated_object_points)

                        # PBW add here, add lidar info
                        object_lidar_info_dict[query_object_token].append(object_lidar_info)
                else:
                    continue
        object_points_dict[query_object_token] = np.concatenate(object_points_dict[query_object_token], axis=0)
        object_lidar_info_dict[query_object_token] = np.concatenate(object_lidar_info_dict[query_object_token], axis=0)


    object_points_vertice = []
    object_lidar_info_vertice = []
    for key in object_points_dict.keys():
        point_cloud = object_points_dict[key]
        object_points_vertice.append(point_cloud[:,:3])

        object_lidar_info = object_lidar_info_dict[key]
        object_lidar_info_vertice.append(object_lidar_info)
    print("whole scene generated!")

    i = 0
    keyframe = 0
    while int(i) < 10000:  # Assuming the sequence does not have more than 10000 frames
        if i >= len(dict_list):
            print('finish scene!')
            return
        dict = dict_list[i]
        is_key_frame = dict['is_key_frame']
        if not is_key_frame: # only use key frame as GT
            i = i + 1
            continue
        
        keyframe += 1
        print(f"deal with {keyframe}th keyframe")
        ################## convert the static scene to the target coordinate system ##############
        lidar_calibrated_sensor = dict['lidar_calibrated_sensor']
        lidar_ego_pose = dict['lidar_ego_pose']
        lidar_pc_i_semantic = lidar_to_world_to_lidar(lidar_pc_with_semantic.copy(),
                                                      lidar_calibrated_sensor0.copy(),
                                                      lidar_ego_pose0.copy(),
                                                      lidar_calibrated_sensor,
                                                      lidar_ego_pose)
        point_cloud_with_semantic = lidar_pc_i_semantic.points.T
        
        ################## load bbox of target frame ##############
        lidar_path, boxes, _ = nusc.get_sample_data(dict['lidar_token'])
        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation.yaw_pitch_roll[0]
                         for b in boxes]).reshape(-1, 1)
        gt_bbox_3d = np.concatenate([locs, dims, rots], axis=1).astype(np.float32)
        gt_bbox_3d[:, 6] += np.pi / 2.
        gt_bbox_3d[:, 2] -= dims[:, 2] / 2.
        gt_bbox_3d[:, 2] = gt_bbox_3d[:, 2] - 0.1
        gt_bbox_3d[:, 3:6] = gt_bbox_3d[:, 3:6] * 1.1
        rots = gt_bbox_3d[:,6:7]
        locs = gt_bbox_3d[:,0:3]

        ################## bbox placement ##############
        object_points_list = []
        object_semantic_list = []
        object_info_list = []
        for j, object_token in enumerate(dict['object_tokens']):
            for k, object_token_in_zoo in enumerate(object_token_zoo):
                if object_token==object_token_in_zoo:
                    points = object_points_vertice[k]
                    points_info = object_lidar_info_vertice[k]
                    Rot = Rotation.from_euler('z', rots[j], degrees=False)
                    rotated_object_points = Rot.apply(points)
                    points = rotated_object_points + locs[j]
                    if points.shape[0] >= 5:
                        points_in_boxes = points_in_boxes_cpu(torch.from_numpy(points[:, :3][np.newaxis, :, :]),
                                                              torch.from_numpy(gt_bbox_3d[j:j+1][np.newaxis, :]))
                        points = points[points_in_boxes[0,:,0].bool()]
                        points_info = points_info[points_in_boxes[0,:,0].bool()]

                    object_points_list.append(points)
                    object_info_list.append(points_info)
                    semantics = np.ones_like(points[:,0:1]) * object_semantic[k]
                    object_semantic_list.append(np.concatenate([points[:, :3], semantics], axis=1))

        try:
            temp = np.concatenate(object_semantic_list)
            scene_semantic_points = np.concatenate([point_cloud_with_semantic, temp])
            object_info = np.concatenate(object_info_list)
            lidar_info = np.concatenate([lidar_info_static, object_info], axis=0)
        except:
            scene_semantic_points = point_cloud_with_semantic
            lidar_info = lidar_info_static

        ################ PBW add here: change lidar coords to vehicle coords #############
        scene_semantic_points = lidar2vehicle(scene_semantic_points, lidar_calibrated_sensor)
        scene_semantic_points = scene_semantic_points.points.T

        ################## remain points with a spatial range (vehicle coords)##############
        mask = (np.abs(scene_semantic_points[:, 0]) < 40.0) & (np.abs(scene_semantic_points[:, 1]) < 40.0) \
               & (scene_semantic_points[:, 2] > -1.0) & (scene_semantic_points[:, 2] < 5.4)
        scene_semantic_points = scene_semantic_points[mask]
        lidar_info = lidar_info[mask]

        ###################### convert points to voxels, generate save infos ######################
        pcd_np = scene_semantic_points[:, :3]
        pcd_np[:, 0] = (pcd_np[:, 0] - pc_range[0]) / voxel_size
        pcd_np[:, 1] = (pcd_np[:, 1] - pc_range[1]) / voxel_size
        pcd_np[:, 2] = (pcd_np[:, 2] - pc_range[2]) / voxel_size
        pcd_np = np.floor(pcd_np).astype(np.int32)

        occ_mask = np.zeros(occ_size).astype(np.uint8) + 255
        occ_index, inverse_indices = np.unique(pcd_np, axis=0, return_inverse=True)
        occ_label = np.zeros(occ_index.shape[0], dtype=np.uint8)

        # for each voxel, vote the label
        counts = np.zeros((occ_index.shape[0], 256), dtype=np.int16)  # 256 用于所有可能的标签
        np.add.at(counts, (inverse_indices, lidar_info), 1)
        occ_label = np.argmax(counts, axis=1)

        occ_mask[tuple(occ_index.T)] = occ_label

        ###################### save img_index_map, occ_mask, voxel_infos to pickle ######################
        save = {}
        save['ovo_gt'] = occ_mask

        dirs = os.path.join(save_path, dict['scene_name'], dict['occ_token'])
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        final_path = os.path.join(dirs, 'label_ovo.pkl')
        with open(final_path, 'wb') as f:
            pickle.dump(save, f)

        i = i + 1
        continue

if __name__ == '__main__':
    from argparse import ArgumentParser
    parse = ArgumentParser()

    parse.add_argument('--dataset', type=str, default='nuscenes')
    parse.add_argument('--config_path', type=str, default='./config.yaml')
    parse.add_argument('--split', type=str, default='train')
    parse.add_argument('--save_path', type=str, default='data/occ3d/san_gts_qwen_scene')
    parse.add_argument('--seg_root', type=str, default='data/occ3d/san_qwen_scene')
    parse.add_argument('--start', type=int, default=0)
    parse.add_argument('--end', type=int, default=750)
    parse.add_argument('--data_root', type=str, default='data/occ3d')
    parse.add_argument('--nusc_val_list', type=str, default='./nuscenes_val_list.txt')
    parse.add_argument('--label_mapping', type=str, default='./nuscenes.yaml')
    args=parse.parse_args()

    if args.dataset=='nuscenes':
        val_list = []
        with open(args.nusc_val_list, 'r') as file:
            for item in file:
                val_list.append(item[:-1])
        file.close()

        nusc = NuScenes(version='v1.0-trainval',
                        dataroot=args.data_root,
                        verbose=True)
        train_scenes = splits.train
        val_scenes = splits.val
    else:
        raise NotImplementedError

    # load config
    with open(args.config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    # load learning map
    label_mapping = args.label_mapping
    with open(label_mapping, 'r') as stream:
        nuscenesyaml = yaml.safe_load(stream)

    for i in tqdm(range(args.start,args.end)):
        print('processing sequecne:', i)
        main(nusc, val_list, indice=i,
             nuscenesyaml=nuscenesyaml, args=args, config=config)