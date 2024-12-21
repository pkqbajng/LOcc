import json
import torch
import torch.nn as nn
import numpy as np
from mmcv.runner import BaseModule
from mmdet3d.models.builder import HEADS
from mmcv.cnn import build_conv_layer, build_norm_layer

def load_text_embedding(text_embedding_file):
    with open(text_embedding_file, 'r') as f1:
        info = json.load(f1)
    k_word_tokens = []
    for k in info:
        if k != 'sky':
            k_word_tokens.append(torch.Tensor(info[k]).unsqueeze(0))
    k_word_tokens = torch.cat(k_word_tokens)
    return k_word_tokens

@HEADS.register_module()
class OVOHeadFeat(BaseModule):
    def __init__(
        self,
        in_dim=32,
        mid_channels_occ=128,
        mid_channels_language=128,
        language_channels=256,
        text_embedding='odise.json',
        num_classes=18,
        mapping_table=None,
        use_mask=False,
        scene_specific=False,
        balance_cls_weight=False,
        conv_cfg=dict(type='Conv3d', bias=False),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),      
    ):
        super(OVOHeadFeat, self).__init__()

        self.occ_head = nn.Sequential(
            build_conv_layer(conv_cfg, in_channels=in_dim,
                    out_channels=mid_channels_occ, kernel_size=3, stride=1, padding=1),
            build_norm_layer(norm_cfg, mid_channels_occ)[1],
            nn.ReLU(inplace=True),
            build_conv_layer(conv_cfg, in_channels=mid_channels_occ, 
                    out_channels=2, kernel_size=1, stride=1, padding=0),
        )

        self.language_head = nn.Sequential(
            build_conv_layer(conv_cfg, in_channels=in_dim, 
                    out_channels=mid_channels_language, kernel_size=3, stride=1, padding=1),
            build_norm_layer(norm_cfg, mid_channels_language)[1],
            nn.ReLU(inplace=True),
            build_conv_layer(conv_cfg, in_channels=mid_channels_language, 
                    out_channels=language_channels, kernel_size=1, stride=1, padding=0),
        )

        self.text_embedding = load_text_embedding(text_embedding)

        self.num_classes = num_classes
        self.mapping_table = mapping_table
        self.use_mask = use_mask
        self.scene_specific = scene_specific
        self.balance_cls_weight = balance_cls_weight
    
    def forward(self, img_feats):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx)

        Returns:

        """

        occ_field = self.occ_head(img_feats).permute(0, 4, 3, 2, 1)
        language_field = self.language_head(img_feats).permute(0, 4, 3, 2, 1)

        output_dict = {
            'occ_field': occ_field,
            'language_field': language_field
        }

        return output_dict
    
    def loss_occ_field(self, occ_field, occ_gt_field, mask=None):
        criterion = nn.CrossEntropyLoss(
            reduction="mean"
        )

        if mask is None:
            loss_occ_field = criterion(occ_field.permute(0, 4, 1, 2, 3), occ_gt_field.long())
        else:
            occ_field_values = occ_field[mask.bool(), :]
            occ_gt_field_values = occ_gt_field[mask.bool()]
            loss_occ_field = criterion(occ_field_values, occ_gt_field_values)

        return loss_occ_field
    
    def loss_language_field(self, language_field, ovo_gt, mask, img_metas=None):
        criterion = nn.CosineSimilarity(
            dim=1, eps=1e-6
        )

        loss_language_field = criterion(
            language_field[mask.bool(), :], ovo_gt[mask.bool(), :]
        )

        loss = torch.sum(1 - loss_language_field) / len(loss_language_field)
        
        return loss
    
    def loss(self, output_dict, gt_dict):
        losses = dict()

        img_metas = gt_dict['img_metas']

        voxel_semantics = gt_dict['voxel_semantics']
        occ_gt_field = (voxel_semantics != 17).to(torch.uint8)
        
        if self.use_mask:
            loss_occ_field = self.loss_occ_field(output_dict['occ_field'], occ_gt_field, mask=gt_dict['mask_camera'])
        else:
            loss_occ_field = self.loss_occ_field(output_dict['occ_field'], occ_gt_field)
        
        ovo_gt_mask = gt_dict['ovo_gt_mask']
        ovo_gt = gt_dict['ovo_gt']

        new_ovo_gt = torch.zeros((1, 200, 200, 16, 512)).to(ovo_gt.device)
        new_ovo_gt[ovo_gt_mask.bool(), :] = ovo_gt.view(-1, 512).to(torch.float32)
        language_mask = ovo_gt_mask * occ_gt_field
        
        if self.use_mask:
            language_mask = language_mask * gt_dict['mask_camera']
        
        loss_language_field = self.loss_language_field(
            output_dict['language_field'], new_ovo_gt, language_mask, img_metas=img_metas
        )

        losses['loss_occ_field'] = loss_occ_field
        losses['loss_language_field'] = loss_language_field
        return losses
    
    def get_occ_pred(self, output_dict):
        with torch.no_grad():
            occ_field = output_dict['occ_field']
            language_field = output_dict['language_field']

            b, h, w, z, _ = language_field.shape
            occ_pred = torch.ones(b, h, w, z, device=occ_field.device, dtype=torch.uint8) * (self.num_classes - 1)
            for i in range(b):
                valid_region = torch.argmax(occ_field[i], dim=-1)
                valid_language = language_field[i][valid_region.bool()]
                valid_classes = torch.einsum("lc,nc->ln", valid_language, self.text_embedding.to(occ_field.device))
                valid_classes = torch.argmax(valid_classes, dim=-1)
                for key, value in self.mapping_table.items():
                    valid_classes[valid_classes == key] = value
                occ_pred[i][valid_region.bool()] = valid_classes.to(torch.uint8)
        
        return occ_pred