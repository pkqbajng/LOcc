import os
import cv2
import torch
import pickle
import numpy as np
from tqdm import tqdm
from nuscenes.utils import splits
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes

def get_open_seg_result(proj_img_index, proj_uv, filenames, args):
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

def get_grid_coords(dims, resolution, center=True):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0]) # [0, 1, ..., 256]
    # g_xx = g_xx[::-1]
    g_yy = np.arange(0, dims[1]) # [0, 1, ..., 256]
    # g_yy = g_yy[::-1]
    g_zz = np.arange(0, dims[2]) # [0, 1, ..., 32]

    # Obtaining the grid with coords...
    yy, xx, zz = np.meshgrid(g_xx, g_yy, g_zz) # 注意这里的顺序！！！！
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)
    resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])
    if center:
        coords_grid = (coords_grid * resolution) + resolution / 2
    else:
        coords_grid = (coords_grid * resolution)

    return coords_grid

if __name__ == '__main__':
    from argparse import ArgumentParser
    parse = ArgumentParser()
    parse.add_argument('--split', type=str, default='train')
    parse.add_argument('--data_root', type=str, default='data/occ3d')
    parse.add_argument('--save_path', type=str, default='data/occ3d/san_gts_projection')
    parse.add_argument('--seg_root', type=str, default='data/occ3d/san_qwen_scene')
    parse.add_argument('--start', type=int, default=0)
    parse.add_argument('--end', type=int, default=10)

    args = parse.parse_args()

    val_list = []
    with open('gt_generation/nuscenes_val_list.txt', 'r') as file:
        for item in file:
            val_list.append(item[:-1])
    file.close()

    nusc = NuScenes(version='v1.0-trainval',
            dataroot=args.data_root, verbose=True)
    
    train_scenes = splits.train
    val_scenes = splits.val
    
    camera_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_FRONT_RIGHT']

    for i in tqdm(range(args.start, args.end)):
        my_scene = nusc.scene[i]

        if args.split == 'train':
            if my_scene['token'] in val_list:
                continue
        
        # load the first sample to start
        first_sample_token = my_scene['first_sample_token']
        my_sample = nusc.get('sample', first_sample_token)
        lidar_data = nusc.get('sample_data', my_sample['data']['LIDAR_TOP'])
        # ego_pose: ego to global; calibrated_sensor: lidar to ego
        lidar_ego_pose0 = nusc.get('ego_pose', lidar_data['ego_pose_token'])
        lidar_calibrated_sensor0 = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        
        voxels = np.zeros((200, 200, 16))
        voxel_size=[0.4, 0.4, 0.4]
        vox_origin = [-40, -40, -1.0]

        grid_coords = get_grid_coords(
            [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
        ) + np.array(vox_origin, dtype=np.float32).reshape([1, 3])
        print("grid shape", grid_coords.shape)
        my_scene_name = my_scene['name']
        print("scene name: ", my_scene_name)

        while True:
            if lidar_data['is_key_frame']:
                index = lidar_data['sample_token']
                rec = nusc.get('sample', index)
                
                scene_name = nusc.get('scene', rec['scene_token'])['name']
                occ_gt_file = os.path.join(args.data_root, 'gts', scene_name, index, 'labels.npz')

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
                
                occ_gt = np.load(occ_gt_file)
                mask_camera = occ_gt['mask_camera']
                semantics = occ_gt['semantics']
                valid_mask = (semantics!=17) * (semantics!=255)
                valid_coords = grid_coords[valid_mask.reshape(-1), :]
                proj_info, proj_mask = vehicle2img(valid_coords, proj_matrix)
                proj_mask = torch.from_numpy(proj_mask)
                proj_uv = proj_info[:, 1:]
                proj_img_index = proj_info[:, 0].astype(np.int16)
                semantic_category = get_open_seg_result(proj_img_index, proj_uv, filenames, args)

                sparse_ovo_gt = np.ones_like(semantics) * 255
                sparse_ovo_gt[valid_mask] = semantic_category

                save = {}
                save['ovo_gt'] = sparse_ovo_gt

                dst_root = os.path.join(args.save_path, scene_name, index)
                if not os.path.exists(dst_root):
                    os.makedirs(dst_root, exist_ok=True)
                
                final_path = os.path.join(dst_root, 'label_ovo.pkl')
                with open(final_path, 'wb') as f:
                    pickle.dump(save, f)

                next_token = lidar_data['next']
                if next_token != '':
                    lidar_data = nusc.get('sample_data', next_token)
                else:
                    break
            else:
                next_token = lidar_data['next']
                if next_token != '':
                    lidar_data = nusc.get('sample_data', next_token)
                else:
                    break