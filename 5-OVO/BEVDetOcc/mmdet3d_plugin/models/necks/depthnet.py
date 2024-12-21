import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from torch.cuda.amp.autocast_mode import autocast
from torch.utils.checkpoint import checkpoint
from mmcv.runner import BaseModule, force_fp32
from mmdet3d.models.builder import NECKS

@NECKS.register_module()
class DepthNet(BaseModule):
    def __init__(self,
                 in_channels,
                 context_channels,
                 grid_config=None,
                 loss_depth_weight=3.0,
                 downsample=8,
                 sid=False,
                 **kwargs
        ):
        super(DepthNet, self).__init__()
        
        self.sid = sid
        self.grid_config = grid_config
        self.downsample = downsample
        self.loss_depth_weight = loss_depth_weight

        depth_channels = int((grid_config['depth'][1] - grid_config['depth'][0]) / grid_config['depth'][2])
        self.D = depth_channels
        self.out_channels = context_channels

        self.conv = nn.Conv2d(in_channels, self.D + context_channels, kernel_size=1, padding=0)
    
    def forward(self, x):
        
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.conv(x)

        depth_digit = x[:, :self.D, ...]    # (B*N, D, fH, fW)
        context = x[:, self.D:self.D + self.out_channels, ...]    # (B*N, C, fH, fW)

        context = context.view(B, N, -1, H, W)
        depth = depth_digit.softmax(dim=1)

        return depth, context

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N, H // self.downsample,
                                   self.downsample, W // self.downsample,
                                   self.downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample,
                                   W // self.downsample)

        if not self.sid:
            gt_depths = (gt_depths - (self.grid_config['depth'][0] -
                                      self.grid_config['depth'][2])) / \
                        self.grid_config['depth'][2]
        else:
            gt_depths = torch.log(gt_depths) - torch.log(
                torch.tensor(self.grid_config['depth'][0]).float())
            gt_depths = gt_depths * (self.D - 1) / torch.log(
                torch.tensor(self.grid_config['depth'][1] - 1.).float() /
                self.grid_config['depth'][0])
            gt_depths = gt_depths + 1.
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(
            gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:,
                                                                           1:]
        return gt_depths.float()
    
    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds):
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3,
                                          1).contiguous().view(-1, self.D)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds,
                depth_labels,
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum())
        return self.loss_depth_weight * depth_loss