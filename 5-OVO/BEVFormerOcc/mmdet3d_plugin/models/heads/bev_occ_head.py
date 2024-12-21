import torch
from torch import nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
import numpy as np
from mmdet3d.models.builder import HEADS, build_loss
from mmcv.runner import force_fp32, auto_fp16

@HEADS.register_module()
class OccHead(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        out_dim=32,
        pillar_h=16,
        num_classes=18,
        loss_occ=None,
        use_camera_mask=True,
        **kwargs
    ):
        super().__init__()
        self.pillar_h = pillar_h
        self.embed_dims = embed_dims
        self.out_dim = out_dim
        self.num_classes = num_classes
        self.use_camera_mask = use_camera_mask
        self.loss_occ = build_loss(loss_occ)
        
        self.decoder = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims * 2),
            nn.Softplus(),
            nn.Linear(self.embed_dims * 2, self.embed_dims * 2),
            )

        self.predicter = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim*2),
            nn.Softplus(),
            nn.Linear(self.out_dim*2,num_classes),
        )

        self.loss_occ = build_loss(loss_occ)
    
    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, bev_embed):
        bs, c, bev_h, bev_w = bev_embed.shape

        outputs = self.decoder(bev_embed.permute(0, 3, 2, 1))
        outputs = outputs.view(bs, bev_w, bev_h, self.pillar_h, self.out_dim)
        outputs = self.predicter(outputs)

        return outputs

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, voxel_semantics,
             occ,
             mask_camera=None,
             mask_lidar=None,
             **kwargs):
        '''
        Loss function. 
        Args:
            voxel_semantics (torch.Tensor): Shape (bs, w, h, total_z)
            valid_mask (torch.Tensor): 1 represent valid voxel, 0 represent invalid voxel. 
                                       Directly get from the data loader. shape (bs, w, h, total_z)
            preds_dicts (dict): result from head with keys "bev_embed, occ, extra".
            - occ (torch.Tensor): Predicted occupancy features with shape (bs, w, h, total_z, c). 
        Returns:
            loss_dict (dict): Losses of different branch. 
                              Default cvtocc model has refine_feat_loss loss and loss_occ_coheam loss. 
        '''

        loss_dict = dict()
        occ = occ.clone()
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= self.num_classes-1, "semantic gt out of range"
        losses = self.loss_single(voxel_semantics, mask_camera, occ)
        loss_dict['loss_occ'] = losses

        return loss_dict

    def loss_single(self, voxel_semantics, mask_camera, preds_dicts):
        if self.use_camera_mask:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds_dicts = preds_dicts.reshape(-1,self.num_classes)
            mask_camera = mask_camera.reshape(-1)
            num_total_samples = mask_camera.sum()
            loss_occ = self.loss_occ(preds_dicts,
                                     voxel_semantics,
                                     mask_camera, 
                                     avg_factor=num_total_samples)
            
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds_dicts = preds_dicts.reshape(-1, self.num_classes)
            pos_num = voxel_semantics.shape[0]

            loss_occ = self.loss_occ(preds_dicts, voxel_semantics.long(), avg_factor=pos_num)

        return loss_occ
    
    @force_fp32(apply_to=('preds_dicts'))
    def get_occ(self, occ):
        """
        Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
        Returns:
            occ_out (torch.Tensor): Predicted occupancy map with shape (bs, h, w, z).
        """

        occ = occ.softmax(-1)
        occ = occ.argmax(-1)

        return occ