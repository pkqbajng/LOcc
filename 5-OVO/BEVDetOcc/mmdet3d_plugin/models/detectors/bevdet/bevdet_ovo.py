import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32, BaseModule

from mmdet3d.models import DETECTORS
from mmdet3d.models import builder

@DETECTORS.register_module()
class BEVDetOVO(BaseModule):
    def __init__(
        self,
        img_backbone,
        img_neck,
        depth_net,
        img_view_transformer,
        img_bev_encoder_backbone,
        img_bev_encoder_neck,
        occ_head=None,
        **kwargs
    ):
        super().__init__()

        self.img_backbone = builder.build_backbone(img_backbone)
        self.img_neck = builder.build_neck(img_neck)

        self.depth_net = builder.build_neck(depth_net)
        self.img_view_transformer = builder.build_neck(img_view_transformer)
        self.img_bev_encoder_backbone = builder.build_backbone(img_bev_encoder_backbone)
        self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck)

        if occ_head is not None:
            self.occ_head = builder.build_head(occ_head)
        
    def image_encoder(self, img):
        """
        Args:
            img: (B, N, 3, H, W)
            stereo: bool
        Returns:
            x: (B, N, C, fH, fW)
            stereo_feat: (B*N, C_stereo, fH_stereo, fW_stereo) / None
        """
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = self.img_backbone(imgs)
        if hasattr(self, 'img_neck'):
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x
    
    def prepare_inputs(self, inputs):
        # split the inputs into each frame
        assert len(inputs) == 7
        B, N, C, H, W = inputs[0].shape
        imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
            inputs

        sensor2egos = sensor2egos.view(B, N, 4, 4)
        ego2globals = ego2globals.view(B, N, 4, 4)

        # calculate the transformation from adj sensor to key ego
        keyego2global = ego2globals[:, 0,  ...].unsqueeze(1)    # (B, 1, 4, 4)
        global2keyego = torch.inverse(keyego2global.double())   # (B, 1, 4, 4)
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()     # (B, N_views, 4, 4)
        sensor2keyegos = sensor2keyegos.float()

        return [imgs, sensor2keyegos, ego2globals, intrins,
                post_rots, post_trans, bda]
    
    @force_fp32()
    def bev_encoder(self, x):
        """
        Args:
            x: (B, C, Dy, Dx)
        Returns:
            x: (B, C', 2*Dy, 2*Dx)
        """
        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)
        if type(x) in [list, tuple]:
            x = x[0]
        return x
    
    def extract_img_feat(self, img_inputs, img_metas, **kwargs):
        imgs, sensor2ego, ego2global, intrin, post_rot, post_tran, bda = img_inputs
        x = self.image_encoder(imgs)

        # mlp_input = self.depth_net.get_mlp_input(sensor2ego, ego2global, intrin, post_rot, post_tran, bda)
        depth, context = self.depth_net(x)
        x = self.img_view_transformer([context] + img_inputs[1:7] + [depth])
        
        x = self.bev_encoder(x)
        return x, depth

    def forward_train(
        self, 
        img_inputs=None, 
        img_metas=None, 
        **kwargs):

        voxel_semantics = kwargs['voxel_semantics']
        mask_camera = kwargs['mask_camera']
        gt_depth = kwargs['gt_depth']

        gt_dict = {
            'voxel_semantics': voxel_semantics,
            'mask_camera': mask_camera,
            'ovo_gt': kwargs['ovo_gt'],
            'ovo_gt_mask': kwargs['ovo_gt_mask'],
            'img_metas': img_metas  
        }

        img_inputs = self.prepare_inputs(img_inputs)
        x, depth = self.extract_img_feat(img_inputs, img_metas, **kwargs)
        
        outs = self.occ_head(x)

        losses = dict()
        # loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        loss_occ = self.occ_head.loss(
            outs, gt_dict)

        losses.update(loss_occ)
        # losses['loss_depth'] = loss_depth
        
        occ_pred = self.occ_head.get_occ_pred(outs)
        
        results_dict = dict()
        results_dict['occ_pred'] = occ_pred
        results_dict['voxel_semantics'] = voxel_semantics
        results_dict['mask_camera'] = mask_camera
        results_dict['losses'] = losses

        return results_dict
    
    def forward_test(
        self, 
        img_inputs=None, 
        img_metas=None, 
        **kwargs):

        voxel_semantics = kwargs['voxel_semantics']
        mask_camera = kwargs['mask_camera']

        img_inputs = self.prepare_inputs(img_inputs)
        x, depth = self.extract_img_feat(img_inputs, img_metas, **kwargs)
        
        outs = self.occ_head(x)
        occ_pred = self.occ_head.get_occ_pred(outs)

        results_dict = dict()
        results_dict['occ_pred'] = occ_pred
        results_dict['voxel_semantics'] = voxel_semantics
        results_dict['mask_camera'] = mask_camera
        results_dict['img_metas'] = img_metas.data

        return results_dict
    
    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
