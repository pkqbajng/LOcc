import torch
import cv2
import numpy as np

from mmdet3d.datasets import DATASETS
from .nuscenes_dataset_bevdet import NuScenesDatasetBEVDet as NuScenesDataset


colors_map = np.array(
    [
        [0,   0,   0, 255],  # 0 undefined
        [255, 158, 0, 255],  # 1 car  orange
        [0, 0, 230, 255],    # 2 pedestrian  Blue
        [47, 79, 79, 255],   # 3 sign  Darkslategrey
        [220, 20, 60, 255],  # 4 CYCLIST  Crimson
        [255, 69, 0, 255],   # 5 traiffic_light  Orangered
        [255, 140, 0, 255],  # 6 pole  Darkorange
        [233, 150, 70, 255], # 7 construction_cone  Darksalmon
        [255, 61, 99, 255],  # 8 bycycle  Red
        [112, 128, 144, 255],# 9 motorcycle  Slategrey
        [222, 184, 135, 255],# 10 building Burlywood
        [0, 175, 0, 255],    # 11 vegetation  Green
        [165, 42, 42, 255],  # 12 trunk  nuTonomy green
        [0, 207, 191, 255],  # 13 curb, road, lane_marker, other_ground
        [75, 0, 75, 255], # 14 walkable, sidewalk
        [255, 0, 0, 255], # 15 unobsrvd
        [0, 0, 0, 0],  # 16 undefined
        [0, 0, 0, 0],  # 16 undefined
    ])


@DATASETS.register_module()
class NuScenesDatasetOccpancy(NuScenesDataset):
    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        input_dict = super(NuScenesDatasetOccpancy, self).get_data_info(index)

        input_dict['occ_gt_path'] = self.data_infos[index]['occ_path']
        return input_dict

    # def vis_occ(self, semantics):
    #     # simple visualization of result in BEV
    #     semantics_valid = np.logical_not(semantics == 17)
    #     d = np.arange(16).reshape(1, 1, 16)
    #     d = np.repeat(d, 200, axis=0)
    #     d = np.repeat(d, 200, axis=1).astype(np.float32)
    #     d = d * semantics_valid
    #     selected = np.argmax(d, axis=2)

    #     selected_torch = torch.from_numpy(selected)
    #     semantics_torch = torch.from_numpy(semantics)

    #     occ_bev_torch = torch.gather(semantics_torch, dim=2,
    #                                  index=selected_torch.unsqueeze(-1))
    #     occ_bev = occ_bev_torch.numpy()

    #     occ_bev = occ_bev.flatten().astype(np.int32)
    #     occ_bev_vis = colors_map[occ_bev].astype(np.uint8)
    #     occ_bev_vis = occ_bev_vis.reshape(200, 200, 4)[::-1, ::-1, :3]
    #     occ_bev_vis = cv2.resize(occ_bev_vis,(400,400))
    #     return occ_bev_vis
