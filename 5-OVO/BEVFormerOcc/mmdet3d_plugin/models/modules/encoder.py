
# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

from turtle import pos
from .custom_base_transformer_layer import MyCustomBaseTransformerLayer
import copy
import warnings
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import force_fp32, auto_fp16
import numpy as np
import torch
import cv2 as cv
import mmcv
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.utils import ext_loader
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class BEVFormerEncoder(TransformerLayerSequence):

    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, 
                 bev_h=200, 
                 bev_w=200,
                 pc_range=None, 
                 num_points_in_pillar=4, 
                 return_intermediate=False,
                 resize=False,
                 **kwargs):

        super(BEVFormerEncoder, self).__init__(*args, **kwargs)
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.return_intermediate = return_intermediate
        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.resize = resize

    @staticmethod
    def get_reference_points(H, W, Z, 
                             num_points_in_pillar, 
                             dim, 
                             bs=1, 
                             device='cuda', 
                             dtype=torch.float):
        """
        Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar. only used if dim == '3d'.
            num_points_in_pillar: number of points in a pillar. only used if dim == '3d'.
        Returns: controlled by flag `dim`
            ref_3d (Tensor): If dim==`3d`, this function return 3d reference points used in `SCA`. 
                             It has shape (bs, num_points_in_pillar, h*w, 3).
            ref_2d (Tensor): If dim==`2d`, this function return 2d reference points used in `TSA`.
                             It has shape (bs, h*w, 1, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d.unsqueeze(0).repeat(bs, 1, 1, 1) # shape: (bs,num_points_in_pillar,h*w,3)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device),
                indexing='ij'
            )
            ref_y = ref_y.reshape(-1).unsqueeze(0) / H
            ref_x = ref_x.reshape(-1).unsqueeze(0) / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    @force_fp32(apply_to=('reference_points', 'img_metas'))
    def point_sampling(self, reference_points, pc_range,  img_metas):
        """
        This method performs point sampling by converting reference points from a 3D coordinate system to a 2D BEV (Bird's Eye View) coordinate system.
        It takes 3d reference points(ref_3d), point_cloud_range(pc_range), and img_metas as inputs, 
        and it returns sampled reference points in the BEV coordinate system (reference_points_cam) and a binary mask indicating valid BEV points (bev_mask).
        Args:
            reference_points (Tensor): 3d reference points with shape (bs, num_points_in_pillar, h*w, 3).
            pc_range (List): [x1, y1, z1, x2, y2, z2], the range of point cloud.
            img_metas (list[dict]): current img meta info. The list has length of batch size.
        Returns:
            reference_points_cam (Tensor): projected reference points in the camera coordinate system with shape (num_cam, bs, h*w, num_points_in_pillar, 2).
            bev_mask (Tensor): binary mask indicating valid points in `reference_points_cam` with shape (num_cam, bs, h*w, num_points_in_pillar).
        """

        # Step 1: prepare `ego2lidar` and `lidar2img` transformation matrix
        ego2lidar = img_metas[0]['ego2lidar']
        ego2lidar = reference_points.new_tensor(ego2lidar)
        lidar2img = [img_meta['lidar2img'] for img_meta in img_metas]
        lidar2img = reference_points.new_tensor(np.asarray(lidar2img))  # (bs, num_cam, 4, 4)
        
        # Step 2: denormalize the reference points(convert it into the ego system coordinate)
        reference_points = reference_points.clone()
        reference_points[..., 0:1] = reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
        reference_points = reference_points.permute(1, 0, 2, 3) # shape: (num_points_in_pillar, bs, h*w, 4)
        
        # Step 3: reshape transform matrix and reference points
        num_points_in_pillar, bs, num_query_HW, _ = reference_points.size()
        num_cam = lidar2img.size(1)
        reference_points = reference_points.unsqueeze(2).repeat(1, 1, num_cam, 1, 1) # (num_points_in_pillar, bs, num_cam, h*w, 4)
        reference_points = reference_points.unsqueeze(-1) # (num_points_in_pillar, bs, num_cam, h*w, 4, 1)
        lidar2img = lidar2img.view(1, bs, num_cam, 1, 4, 4).repeat(num_points_in_pillar, 1, 1, num_query_HW, 1, 1)
        ego2lidar = ego2lidar.view(1,1,1,1,4,4).repeat(num_points_in_pillar, bs, num_cam, num_query_HW, 1, 1)

        # Step 4: project the reference points to the image plane
        lidar2img = lidar2img.to(torch.float32)
        ego2lidar = ego2lidar.to(torch.float32)
        ego2img = torch.matmul(lidar2img, ego2lidar) # (num_points_in_pillar, bs, num_cam, num_query_HW, 4, 4)
        reference_points = reference_points.to(torch.float32)
        reference_points_img = torch.matmul(ego2img, reference_points)
        reference_points_cam = reference_points_img.squeeze(-1)

        # Step 5: normalize the camera reference points
        eps = 1e-5
        bev_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

        if self.resize:
            post_rot = img_metas[0]['post_rot']
            post_tran = img_metas[0]['post_tran']
            post_rot = post_rot.detach().to(reference_points.device)
            post_tran = post_tran.detach().to(reference_points.device)
            reference_points_cam = post_rot.view(1, 1, 1, 1, 2, 2).matmul(reference_points_cam.unsqueeze(-1)).squeeze(-1)
            reference_points_cam += post_tran.view(1, 1, 1, 1, 2) 

        reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]

        # Step 6: use bev_mask to filter out the points that are outside the image boundary
        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))
        
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(np.nan_to_num(bev_mask.cpu().numpy()))
        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4) # (num_cam, bs, h*w, num_points_in_pillar, 2)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1) # (num_cam, bs, h*w, num_points_in_pillar)

        return reference_points_cam, bev_mask

    @auto_fp16()
    def forward(self,
                bev_query,
                key,
                value,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                img_metas=None,
                shift=None,
                **kwargs):
        """
        Encoder of bevformer, which contains multiple layers. It can construct BEV features from flattened multi level image features.
        Args:
            bev_query (Tensor): Input BEV query with shape (num_query, bs, embed_dims).
            key & value (Tensor): Input multi-camera features with shape (num_cam, num_value, bs, embed_dims). 
            bev_pos (torch.Tensor): bev position embedding with shape (bs, embed_dims, 1, h, w). 
            spatial_shapes (Tensor): spatial shapes of multi-level features.
            level_start_index (Tensor): index of mlvl_feat in all level features
            prev_bev (Tensor): shape (bev_h*bev_w, bs, embed_dims) if use temporal self attention.
            img_metas (list[dict]): current img meta info. The list has length of batch size. 
            shift (Tensor): If `use_can_bus`, the `shift` tensor get from `can_bus` in img_metas. 
                            If not, `shift` tensor is bev_queries.new_zeros((1, 2)). 
        Returns:
            output (Tensor): forwarded results with shape (bs, num_query, embed_dims).

        """

        # Step 1: prepare the reference points. 3d reference points for spatial cross-attention (SCA) and 2d reference points for temporal self-attention (TSA).
        bev_h, bev_w = self.bev_h, self.bev_w
        _dim = 2
        output = bev_query
        intermediate = []
        pc_range_z = self.pc_range[5] - self.pc_range[2]
    
        ref_3d = self.get_reference_points(H=bev_h, W=bev_w, Z=pc_range_z, 
                                           num_points_in_pillar=self.num_points_in_pillar, 
                                           dim='3d',
                                           bs=bev_query.size(1),  
                                           device=bev_query.device, 
                                           dtype=bev_query.dtype)
        # ref_3d: (bs, num_points_in_pillar, h*w, 3)

        ref_2d = self.get_reference_points(H=bev_h, W=bev_w, Z=None,
                                           num_points_in_pillar=None,
                                           dim='2d', 
                                           bs=bev_query.size(1),
                                           device=bev_query.device, 
                                           dtype=bev_query.dtype)
        # ref_2d: (bs, h*w, 1, 2)

        # Step 2: project the 3d reference points to the camera coordinate system and get the binary mask.
        reference_points_cam, bev_mask = self.point_sampling(ref_3d, self.pc_range, img_metas=img_metas)
        # reference_points_cam: (num_cam, bs, h*w, num_points_in_pillar, 2)
        # bev_mask: (num_cam, bs, h*w, num_points_in_pillar)

        # Step 3: prepare the shift reference points for prev BEV features.
        # bug: this code should be 'shift_ref_2d = ref_2d.clone()', we keep this bug for reproducing our results in paper.
        shift_ref_2d = ref_2d  # .clone()
        shift_ref_2d += shift[:, None, None, :]

        # Step 4: reshape the bev_query and bev_pos
        bev_query = bev_query.permute(1, 0, 2)
        if bev_pos is not None: bev_pos = bev_pos.permute(1, 0, 2)

        # Step 5: prepare prev_bev and hybird_ref_2d
        bs, len_bev, num_bev_level, _ = ref_2d.shape # (bs, h*w, 1, 2)
        if prev_bev is not None:
            prev_bev = prev_bev.permute(1, 0, 2)
            prev_bev = torch.stack([prev_bev, bev_query], 1).reshape(bs*2, len_bev, -1)
            hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(bs*2, len_bev, num_bev_level, _dim)
        else:
            hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(bs*2, len_bev, num_bev_level, _dim)

        # Step 6: run the encoder layers
        for layer_idx, layer in enumerate(self.layers):
            output = layer(query=bev_query,
                           key=key,
                           value=value,
                           bev_pos=bev_pos,
                           ref_2d=hybird_ref_2d,
                           ref_3d=ref_3d,
                           spatial_shapes=spatial_shapes,
                           level_start_index=level_start_index,
                           reference_points_cam=reference_points_cam,
                           bev_mask=bev_mask,
                           prev_bev=prev_bev,
                           **kwargs)

            # Step 7: update the input `bev_query` of the next layer according to the output of the current layer
            bev_query = output
            if self.return_intermediate: # Default value is False
                intermediate.append(output)

        if self.return_intermediate: # Default value is False
            return torch.stack(intermediate)

        return output


@TRANSFORMER_LAYER.register_module()
class BEVFormerLayer(MyCustomBaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default:None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default:2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 bev_h=200,
                 bev_w=200,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(BEVFormerLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        self.bev_h = bev_h
        self.bev_w = bev_w

    def forward(self, query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                bev_mask=None,
                prev_bev=None,
                **kwargs):
        """
        Forward function for `TransformerDecoderLayer`.
        Args:
            query (Tensor): The input BEV query with shape (bs, num_queries, embed_dims).
            key (Tensor): The key tensor is flattened multi level image feature with shape (num_cam, num_value, bs, embed_dims). 
            value (Tensor): The value tensor with same shape as `key`.
            bev_pos (Tensor): bev position embedding with shape (bs, embed_dims, 1, h, w). 
            ref_2d (Tensor): hybird 2D reference points used in TSA. 
                             If `prev_bev` is None, it has shape (bs, h*w, 1, 2).
                             else, it has shape (bs*2, h*w, 1, 2).
            ref_3d (Tensor): 3D reference points used in SCA with shape (bs, num_points_in_pillar, h*w, 3).
            reference_points_cam (Tensor): projected reference points in the camera coordinate system with shape (num_cam, bs, h*w, num_points_in_pillar, 2).
            spatial_shapes (Tensor): spatial shapes of multi-level features.
            level_start_index (Tensor): index of mlvl_feat in all level features
            bev_mask (Tensor): binary mask indicating valid points in `reference_points_cam` with shape (num_cam, bs, h*w, num_points_in_pillar).
            prev_bev (Tensor): shape (bs*2, bev_h*bev_w, embed_dims) if use temporal self attention.
            Others are None. 
        Returns:
            query (Tensor): forwarded query results with shape [num_queries, bs, embed_dims].
        """

        # Step 1: prepare the index of the current layer
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query

        # Step 2: prepare the attention masks
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                                                     f'operation_order {self.num_attn}'

        # Step 3: run the encoder layers
        for layer in self.operation_order:
            # Step 4: run the self-attention layer
            if layer == 'self_attn':
                spatial_shapes_tsa = torch.tensor([[self.bev_h, self.bev_w]], device=query.device)
                level_start_index_tsa = torch.tensor([0], device=query.device)
                
                query = self.attentions[attn_index](query=query,
                                                    key=prev_bev,
                                                    value=prev_bev,
                                                    residual=identity if self.pre_norm else None,
                                                    query_pos=bev_pos,
                                                    key_pos=bev_pos,
                                                    attn_mask=attn_masks[attn_index],
                                                    key_padding_mask=query_key_padding_mask,
                                                    reference_points=ref_2d,
                                                    spatial_shapes=spatial_shapes_tsa,
                                                    level_start_index=level_start_index_tsa,
                                                    **kwargs)
                
                attn_index += 1
                identity = query # identity will not go through the normalization layer.

            # There is always a normlization layer after the self-attention layer, cross-attention layer and ffn layer.
            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # Step 5: run the cross-attention layer
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](query,
                                                    key,
                                                    value,
                                                    residual=identity if self.pre_norm else None,
                                                    query_pos=query_pos,
                                                    key_pos=key_pos,
                                                    reference_points=ref_3d,
                                                    reference_points_cam=reference_points_cam,
                                                    bev_mask=bev_mask,
                                                    mask=mask,
                                                    attn_mask=attn_masks[attn_index],
                                                    key_padding_mask=key_padding_mask,
                                                    spatial_shapes=spatial_shapes,
                                                    level_start_index=level_start_index,
                                                    **kwargs)
                
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
