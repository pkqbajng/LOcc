import json
import torch
from torch import nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
import numpy as np
from mmdet3d.models.builder import HEADS, build_loss
from mmcv.runner import force_fp32, auto_fp16

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
class OVOHead(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        mid_channels=64,
        mid_channels_occ=128,
        mid_channels_language=128,
        language_channels=256,
        pillar_h=16,
        num_classes=18,
        loss_occ=None,
        use_camera_mask=True,
        mapping_table=None,
        text_embedding='odise.json',
        scene_specific=False,
        **kwargs
    ):
        super().__init__()
        self.pillar_h = pillar_h
        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.use_camera_mask = use_camera_mask
        self.loss_occ = build_loss(loss_occ)
        self.mid_channels = mid_channels
        self.scene_specific = scene_specific
        self.decoder = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims * 2),
            nn.Softplus(),
            nn.Linear(self.embed_dims * 2, pillar_h * mid_channels),
            )

        self.occ_head = nn.Sequential(
            nn.Linear(self.mid_channels, mid_channels_occ),
            nn.Softplus(),
            nn.Linear(mid_channels_occ,2),
        )

        self.language_head = nn.Sequential(
            nn.Linear(self.mid_channels, mid_channels_language),
            nn.Softplus(),
            nn.Linear(mid_channels_language, language_channels),
        )

        self.loss_occ = build_loss(loss_occ)
        self.text_embedding = load_text_embedding(text_embedding)
        self.mapping_table = mapping_table
    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, bev_embed):
        bs, c, bev_h, bev_w = bev_embed.shape

        voxel_embed = self.decoder(bev_embed.permute(0, 3, 2, 1))

        voxel_embed = voxel_embed.view(bs, bev_w, bev_h, self.pillar_h, self.mid_channels)

        occ_field = self.occ_head(voxel_embed)
        language_field = self.language_head(voxel_embed)

        output_dict = {
            'occ_field': occ_field,
            'language_field': language_field
        }

        return output_dict

    def get_occ(self, output_dict):
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
        if self.scene_specific:
            # To do: bs > 1
            text_embedding_file = img_metas[0]['text_embedding_file']
            text_embedding = load_text_embedding(text_embedding_file)
        else:
            text_embedding = self.text_embedding.clone()
        
        indexes = np.int64(ovo_gt[mask.bool()].detach().cpu().numpy())
        language_gt_field = text_embedding[indexes, :].to(language_field.device)

        criterion = nn.CosineSimilarity(
            dim=1, eps=1e-6
        )

        loss_language_field = criterion(
            language_field[mask.bool(), :], language_gt_field
        )

        loss = torch.sum(1 - loss_language_field) / len(loss_language_field)
        return loss
    
    def loss(self, voxel_semantics,
             output_dict,
             mask_camera=None,
             mask_lidar=None,
             **kwargs):
        losses = dict()
        occ_gt_field = (voxel_semantics != 17).to(torch.uint8)
        loss_occ_field = self.loss_occ_field(output_dict['occ_field'], occ_gt_field, mask=mask_camera)
        
        ovo_gt_mask = output_dict['ovo_gt_mask']
        ovo_gt = output_dict['ovo_gt']
        language_mask = ovo_gt_mask * occ_gt_field
        language_mask = language_mask * mask_camera
        img_metas = output_dict['img_metas']
        loss_language_field = self.loss_language_field(
            output_dict['language_field'], ovo_gt, language_mask, img_metas=img_metas
        )

        losses = dict()
        losses['loss_occ_field'] = loss_occ_field
        losses['loss_language_field'] = loss_language_field
        return losses