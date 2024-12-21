# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import torch
import torch.nn as nn
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import BaseModule, force_fp32
from mmdet.models.utils import build_transformer
from mmdet.models.builder import build_loss
from mmdet.models import HEADS

@HEADS.register_module()
class BEVOccHead(BaseModule):
    def __init__(
        self,
        bev_h=200,
        bev_w=200,
        num_classes=18,
        transformer=None,
        **kwargs
    ):
        super(BEVOccHead, self).__init__()

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False
        self.num_classes = num_classes
        
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims
        self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)
    
    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()

    @auto_fp16(apply_to=('multi_level_feats'))
    def forward(self,
                multi_level_feats, 
                cur_img_metas, 
                prev_bev_list=[], 
                prev_img_metas=[],
                only_bev=False,
                **kwargs):
        """
        Forward function.
        Args:
            multi_level_feats (list[torch.Tensor]): Current multi level img features from the upstream network.
                                                    Each is a 5D-tensor img_feats with shape (bs, num_cams, embed_dims, h, w).
            cur_img_metas (list[dict]): Meta information of each sample. The list has length of batch size.
            prev_bev_list (list[torch.Tensor]): BEV features of previous frames. Each has shape (bs, bev_h*bev_w, embed_dims). 
            prev_img_metas (list[dict[dict]]): Meta information of each sample.
                                               The list has length of batch size.
                                               The dict has keys len_queue-1-prev_bev_list_len, ..., len_queue-2. 
                                               The element of each key is a dict.
                                               So each dict has length of prev_bev_list_len. 
            only_bev: If this flag is true. The head only computes BEV features with encoder.
        Returns:
            If only_bev:
            _bev_embed (torch.Tensor): BEV features of the current frame with shape (bs, bev_h*bev_w, embed_dims). 
            else: 
            outs (dict): with keys "bev_embed, occ, extra".
            - bev_embed (torch.Tensor): BEV features of the current frame with shape (bs, bev_h*bev_w, embed_dims).
            - occ (torch.Tensor): Predicted occupancy features with shape (bs, w, h, total_z, c).
            - extra (dict): extra information. if 'costvolume' in self.transformer, it will have 'refine_feat_w' key.
        """

        # Step 1: initialize BEV queries and mask
        bs = multi_level_feats[0].shape[0]
        dtype = multi_level_feats[0].dtype
        bev_queries = self.bev_embedding.weight.to(dtype)
        # bev_mask = torch.zeros((bs, self.bev_h, self.bev_w), device=bev_queries.device).to(dtype)
        bev_pos = None

        # Step 2: get BEV features
        if only_bev:
            if len(prev_bev_list) == 0:
                prev_bev = None
            else:
                prev_bev = prev_bev_list[-1]

            outputs = self.transformer.get_bev_features(multi_level_feats,
                                                        bev_queries,
                                                        bev_pos,
                                                        cur_img_metas,
                                                        prev_bev,
                                                        **kwargs)
            _bev_embed = outputs['bev_embed']
            
            return _bev_embed

        else:
            bev_embed = self.transformer(multi_level_feats,
                                       bev_queries,
                                       bev_pos,
                                       cur_img_metas,
                                       prev_bev_list,
                                       prev_img_metas,
                                       **kwargs)
            
            return bev_embed