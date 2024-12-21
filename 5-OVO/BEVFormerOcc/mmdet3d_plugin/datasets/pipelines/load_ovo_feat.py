import os
import json
import torch
import numpy as np
from mmdet.datasets.builder import PIPELINES

@PIPELINES.register_module()
class LoadOVOFeatFromFile(object):
    def __init__(
        self,
        ovo_gt_root,
        data_root=None
    ):
        self.ovo_gt_root = ovo_gt_root
        self.data_root = data_root

    def __call__(self, results):
        occ_gt_path = results['occ_gt_path']
        ovo_gt_path = occ_gt_path.replace('gts', self.ovo_gt_root)
        ovo_gt_path = os.path.join(self.data_root, ovo_gt_path)
        # ovo_gt_path = os.path.join(ovo_gt_path, "labels.npz")
        ovo_gt_dict = np.load(ovo_gt_path)
        ovo_gt_feat = ovo_gt_dict['feats']
        voxel_coords = ovo_gt_dict['voxel_coords']

        ovo_gt_mask = np.zeros((200, 200, 16))
        ovo_gt_mask[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]] = 1
        ovo_gt = torch.from_numpy(ovo_gt_feat)
        ovo_gt_mask = torch.from_numpy(ovo_gt_mask)
        
        if results.get('flip_dx', False):
            ovo_gt = torch.flip(ovo_gt, [0])
            ovo_gt_mask = torch.flip(ovo_gt_mask, [0])
        
        if results.get('flip_dy', False):
            ovo_gt = torch.flip(ovo_gt, [1])
            ovo_gt_mask = torch.flip(ovo_gt_mask, [1])
        
        results['ovo_gt'] = ovo_gt
        results['ovo_gt_mask'] = ovo_gt_mask

        return results