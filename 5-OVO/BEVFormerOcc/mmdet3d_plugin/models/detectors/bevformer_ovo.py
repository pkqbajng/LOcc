import copy
from errno import ESTALE
import numpy as np
import torch
from mmdet3d.models import builder
from mmdet.models import DETECTORS
from mmcv.runner import BaseModule, force_fp32, auto_fp16
from .grid_mask import GridMask

@DETECTORS.register_module()
class BEVFormerOVO(BaseModule):
    def __init__(
        self,
        use_grid_mask=True,
        video_test_mode=True,
        queue_length=1,
        img_backbone=None,
        img_neck=None,
        img_view_transformer=None,
        occ_head=None,
        **kwargs
    ):
        super().__init__()

        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        
        # temporal
        self.queue_length = queue_length
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev_list': [],
            'prev_img_metas_list': [],
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = builder.build_neck(img_neck)
        
        if img_view_transformer:
            self.img_view_transformer = builder.build_head(img_view_transformer)
        
        if occ_head:
            self.occ_head = builder.build_head(occ_head)
    
    @property
    def with_img_backbone(self):
        """bool: Whether the detector has a 2D image backbone."""
        return hasattr(self, 'img_backbone') and self.img_backbone is not None
    
    @property
    def with_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'img_neck') and self.img_neck is not None

    def extract_img_feat(self, img, len_queue=None):
        """Extract features of images.
        Args: 
            img (torch.Tensor): Image tensor with shape (bs, n_views, C, H, W).
                                But for previous img, its shape will be (bs*len_queue, n_views, C, H, W).
            len_queue (int): The length of the queue. It is less or equal to self.queue_length.
                             It is used when extracting features of previous images.
        Returns:
            list[torch.Tensor]: Image features. Each with shape (bs, n_views, C, H, W).
                                But different scales (from FPN) will have different shapes.
                                For previous img, its shape will be (bs, len_queue, n_views, C, H, W).
        """

        bs_length, num_views, C, H, W = img.size()
        bs_length_num_views = bs_length * num_views
        img = img.reshape(bs_length_num_views, C, H, W)

        if self.use_grid_mask:
            img = self.grid_mask(img)
        img_feats = self.img_backbone(img)
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            bs_length_num_views, C, H, W = img_feat.size()
            if len_queue is not None: # for prev imgs
                bs = int(bs_length / len_queue)
                img_feats_reshaped.append(img_feat.view(bs, len_queue, num_views, C, H, W))
            else: # for current imgs
                img_feats_reshaped.append(img_feat.view(bs_length, num_views, C, H, W))

        return img_feats_reshaped

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, len_queue=None):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, len_queue=len_queue)
        return img_feats
    
    @auto_fp16(apply_to=('img'))
    def forward_train(self, img=None,
                    voxel_semantics=None,
                    mask_lidar=None,
                    mask_camera=None,
                    img_metas=None,
                    **kwargs):
        
        # Step 1: prepare cur_img_feats and cur_img_metas
        batch_size, len_queue, _, _, _, _ = img.shape
        cur_img = img[:, -1, ...]
        cur_img_feats = self.extract_feat(img=cur_img) # list[tensor], each tensor is of shape (B, N, C, H, W). H and W are different across scales. 
        img_metas_deepcopy = copy.deepcopy(img_metas.data[0])
        cur_img_metas = [each_batch[len_queue-1] for each_batch in img_metas.data[0]] # list[dict] of length equals to batch_size

        # Step 2: prepare prev_bev_list, prev_img_metas
        if cur_img_metas[0]['prev_bev_exists']:
            prev_img = img[:, :-1, ...]
            bs, prev_len_queue, num_cams, C, H, W = prev_img.shape
            prev_img = prev_img.reshape(bs * prev_len_queue, num_cams, C, H, W)
            with torch.no_grad():
                prev_img_feats = self.extract_feat(img=prev_img, len_queue=prev_len_queue)

            prev_img_metas = []
            for each_batch in img_metas_deepcopy:
                each_batch.pop(len_queue - 1)
                prev_img_metas.append(each_batch) # list[dict[dict]]
                
            prev_bev_list = self.obtain_history_bev(prev_img_feats, prev_img_metas, prev_len_queue)

            # Step 3: adjust the length of these two to be consistent
            prev_bev_list_len = len(prev_bev_list)
            for each_batch in prev_img_metas:
                if len(each_batch) > prev_bev_list_len:
                    for i in range(0, len(each_batch) - prev_bev_list_len): # len(each_batch) = len_queue - 1
                        each_batch.pop(i)
        else:
            prev_bev_list = []
            prev_img_metas = [{} for _ in range(batch_size)]

        bev_embed = self.img_view_transformer(
            multi_level_feats=cur_img_feats,
            cur_img_metas=cur_img_metas,
            prev_bev_list=prev_bev_list,
            prev_img_metas=prev_img_metas,
            only_bev=False,
            **kwargs
        )

        outs = self.occ_head(bev_embed)
        
        outs['ovo_gt'] = kwargs['ovo_gt']
        outs['ovo_gt_mask'] = kwargs['ovo_gt_mask']
        outs['img_metas'] = cur_img_metas
        losses_occ = self.occ_head.loss(
            voxel_semantics,
            outs,
            mask_camera,
            mask_lidar,
            **kwargs)
        
        occ = self.occ_head.get_occ(outs)
        losses = dict()
        losses.update(losses_occ)

        output_dict = {}
        output_dict['losses'] = losses
        output_dict['occ_pred'] = occ
        output_dict['voxel_semantics'] = voxel_semantics
        output_dict['mask_camera'] = mask_camera
        return output_dict
    
    def forward_test(self, 
            img=None,
            voxel_semantics=None,
            mask_lidar=None,
            mask_camera=None,
            img_metas=None,
            **kwargs):

        for var, name in [(img_metas.data[0], 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(name, type(var)))
            
        # Step 1: prepare the input
        # all arg are be wrapped one more list, so we need to take the first element
        if img is not None: img = img[0]
        if voxel_semantics is not None: voxel_semantics = voxel_semantics[0]
        if mask_camera is not None: mask_camera = mask_camera[0]
        if mask_lidar is not None: mask_lidar = mask_lidar[0]
        if img_metas is not None: 
            cur_img_metas = [each_batch[0] for each_batch in img_metas.data[0]]
            img_metas = img_metas.data[0][0]

        # If the input frame is in a new scene, the prev_frame_info need to be reset.
        if img_metas[0]['scene_token'] != self.prev_frame_info['scene_token']:
            self.prev_frame_info['prev_bev_list'] = []
            self.prev_frame_info['prev_img_metas_list'] = []
            # update idx
            self.prev_frame_info['scene_token'] = img_metas[0]['scene_token']

        if not self.video_test_mode:
            # defalut value of self.video_test_mode is True
            self.prev_frame_info['prev_bev_list'] = []
            self.prev_frame_info['prev_img_metas_list'] = []

        # Step 2: Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0]['can_bus'][-1])
        if len(self.prev_frame_info['prev_bev_list']) > 0:
            img_metas[0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0]['can_bus'][-1] = 0
            img_metas[0]['can_bus'][:3] = 0

        # Step 3: prepare prev_bev_list, prev_img_metas_list
        if len(self.prev_frame_info['prev_bev_list']) > 0:
            prev_bev_list = self.prev_frame_info['prev_bev_list']
            prev_img_metas_list = self.prev_frame_info['prev_img_metas_list']
        else:
            prev_bev = torch.zeros([1, 40000, 256], device=img.device, dtype=img.dtype)
            prev_bev_list = [prev_bev]
            prev_img_metas_list = [img_metas[0].copy()]

        # convert the list to dict TODO
        prev_img_metas_list_len = len(prev_img_metas_list)
        prev_img_metas_dict = {}
        for i in range(prev_img_metas_list_len):
            prev_img_metas_dict[self.queue_length - 1 - prev_img_metas_list_len + i] = prev_img_metas_list[i]
            # from 0 to self.queue_length - 2

        # Step 4: forward in head to get occ_results
        multi_level_feats = self.extract_feat(img=img)
        bev_embed = self.img_view_transformer(
            multi_level_feats=multi_level_feats,
            cur_img_metas=cur_img_metas,
            prev_bev_list=prev_bev_list,
            prev_img_metas=[prev_img_metas_dict],
            only_bev=False,
            **kwargs
        )

        outs = self.occ_head(bev_embed)
        occ = self.occ_head.get_occ(outs)

        output_dict = {}
        output_dict['occ_pred'] = occ[0]
        output_dict['voxel_semantics'] = voxel_semantics
        output_dict['mask_camera'] = mask_camera
        output_dict['img_metas'] = img_metas
        return output_dict
    
    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)