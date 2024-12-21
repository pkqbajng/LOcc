import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer

@HEADS.register_module()
class plugin_segmentation_head(nn.Module):
    def __init__(
        self,
        in_channels=64,
        out_channel_list=[64, 64, 64],
        language_channels=128,
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        text_embedding='odise.json',
        train_cfg=None,
        test_cfg=None
    ):
        super(plugin_segmentation_head, self).__init__()
        in_channel = in_channels
        self.deconv_blocks = nn.ModuleList()
        for out_channel in out_channel_list:
            self.deconv_blocks.append(
                nn.Sequential(
                build_upsample_layer(
                upsample_cfg,
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=2,
                stride=2),
                build_norm_layer(norm_cfg, out_channel)[1],
                nn.ReLU(inplace=True))
            )
            in_channel = out_channel
        
        self.pred = nn.Conv2d(out_channel_list[-1], language_channels, kernel_size=1, stride=1)

        with open(text_embedding, 'r') as f1:
            info = json.load(f1)
        
        k_word_tokens = []
        for k in info:
            if k != 'sky':
                k_word_tokens.append(torch.Tensor(info[k]).unsqueeze(0))
        k_word_tokens = torch.cat(k_word_tokens)
        self.text_embedding = k_word_tokens

    def forward(self, x):
        if len(x.shape) > 4:
            B, N, C, H, W = x.shape
            feat = x.view(B * N, C, H, W)
        else:
            feat = x.clone()
        
        for deconv_block in self.deconv_blocks:
            feat = deconv_block(feat)
        
        seg_pred = self.pred(feat)

        if len(x.shape) > 4:
            _, C, H, W = seg_pred.shape
            seg_pred = seg_pred.view(B, N, C, H, W)
        
        return seg_pred
    
    def get_language_loss(self, ovo_segs, ovo_seg_masks, language_feat):
        language_feat = language_feat.permute(0, 1, 3, 4, 2)
        indexes = np.int64(ovo_segs[ovo_seg_masks.bool()].detach().cpu().numpy())
        language_gt_field = self.text_embedding[indexes, :].to(language_feat.device)

        criterion = nn.CosineSimilarity(
            dim=1, eps=1e-6)
        
        loss_language_field = criterion(
            language_feat[ovo_seg_masks.bool(), :], language_gt_field)
        
        loss = torch.sum(1 - loss_language_field) / len(loss_language_field)
        return loss