import torch
import torch.nn.functional as F
from mmcv.runner import BaseModule, force_fp32

from mmdet3d.models import DETECTORS
from mmdet3d.models import builder
from mmdet.models.backbones.resnet import ResNet
from mmdet3d.models import ResNet

@DETECTORS.register_module()
class BEVStereo4DOVO(BaseModule):
    def __init__(
        self,
        img_backbone,
        img_neck, 
        depth_net,
        img_view_transformer,
        img_bev_encoder_backbone,
        img_bev_encoder_neck,
        occ_head=None,
        align_after_view_transfromation=False,
        pre_process=None,
        num_adj=1,
        with_prev=True,
        **kwargs
    ):
        super().__init__()

        self.align_after_view_transformation = align_after_view_transfromation
        self.num_frame = num_adj + 1
        self.extra_ref_frames = 1
        self.temporal_frame = self.num_frame
        self.num_frame += self.extra_ref_frames

        self.with_prev = with_prev

        self.img_backbone = builder.build_backbone(img_backbone)
        self.img_neck = builder.build_neck(img_neck)

        self.depth_net = builder.build_neck(depth_net)
        self.img_view_transformer = builder.build_neck(img_view_transformer)
        self.img_bev_encoder_backbone = builder.build_backbone(img_bev_encoder_backbone)
        self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck)

        self.pre_process = pre_process is not None
        if self.pre_process:
            self.pre_process_net = builder.build_backbone(pre_process)

        if occ_head is not None:
            self.occ_head = builder.build_head(occ_head)

    def image_encoder(self, img, stereo=False):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = self.img_backbone(imgs)
        stereo_feat = None
        if stereo:
            stereo_feat = x[0]
            x = x[1:]
        if hasattr(self, 'img_neck'):
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x, stereo_feat
    
    def extract_stereo_ref_feat(self, x):
        B, N, C, imH, imW = x.shape
        x = x.view(B * N, C, imH, imW)
        if isinstance(self.img_backbone,ResNet):
            if self.img_backbone.deep_stem:
                x = self.img_backbone.stem(x)
            else:
                x = self.img_backbone.conv1(x)
                x = self.img_backbone.norm1(x)
                x = self.img_backbone.relu(x)
            x = self.img_backbone.maxpool(x)
            for i, layer_name in enumerate(self.img_backbone.res_layers):
                res_layer = getattr(self.img_backbone, layer_name)
                x = res_layer(x)
                return x
        else:
            x = self.img_backbone.patch_embed(x)
            hw_shape = (self.img_backbone.patch_embed.DH,
                        self.img_backbone.patch_embed.DW)
            if self.img_backbone.use_abs_pos_embed:
                x = x + self.img_backbone.absolute_pos_embed
            x = self.img_backbone.drop_after_pos(x)

            for i, stage in enumerate(self.img_backbone.stages):
                x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
                out = out.view(-1,  *out_hw_shape,
                               self.img_backbone.num_features[i])
                out = out.permute(0, 3, 1, 2).contiguous()
                return out
    
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
    
    def prepare_inputs(self, inputs, stereo=False):
        # split the inputs into each frame
        B, N, C, H, W = inputs[0].shape
        N = N // self.num_frame
        imgs = inputs[0].view(B, N, self.num_frame, C, H, W)
        imgs = torch.split(imgs, split_size_or_sections=1, dim=2)
        imgs = [t.squeeze(2) for t in imgs]
        sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
            inputs[1:7]

        sensor2egos = sensor2egos.view(B, self.num_frame, N, 4, 4)
        ego2globals = ego2globals.view(B, self.num_frame, N, 4, 4)

        # calculate the transformation from sweep sensor to key ego
        keyego2global = ego2globals[:, 0, 0, ...].unsqueeze(1).unsqueeze(1)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()

        curr2adjsensor = None
        if stereo:
            sensor2egos_cv, ego2globals_cv = sensor2egos, ego2globals
            sensor2egos_curr = \
                sensor2egos_cv[:, :self.temporal_frame, ...].double()
            ego2globals_curr = \
                ego2globals_cv[:, :self.temporal_frame, ...].double()
            sensor2egos_adj = \
                sensor2egos_cv[:, 1:self.temporal_frame + 1, ...].double()
            ego2globals_adj = \
                ego2globals_cv[:, 1:self.temporal_frame + 1, ...].double()
            curr2adjsensor = \
                torch.inverse(ego2globals_adj @ sensor2egos_adj) \
                @ ego2globals_curr @ sensor2egos_curr
            curr2adjsensor = curr2adjsensor.float()
            curr2adjsensor = torch.split(curr2adjsensor, 1, 1)
            curr2adjsensor = [p.squeeze(1) for p in curr2adjsensor]
            curr2adjsensor.extend([None for _ in range(self.extra_ref_frames)])
            assert len(curr2adjsensor) == self.num_frame

        extra = [
            sensor2keyegos,
            ego2globals,
            intrins.view(B, self.num_frame, N, 3, 3),
            post_rots.view(B, self.num_frame, N, 3, 3),
            post_trans.view(B, self.num_frame, N, 3)
        ]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        sensor2keyegos, ego2globals, intrins, post_rots, post_trans = extra
        return imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
               bda, curr2adjsensor
    
    def prepare_bev_feat(self, img, sensor2keyego, ego2global, intrin,
            post_rot, post_tran, bda, mlp_input, feat_prev_iv,
            k2s_sensor, extra_ref_frame):
        
        if extra_ref_frame:
            stereo_feat = self.extract_stereo_ref_feat(img)
            return None, None, stereo_feat
        x, stereo_feat = self.image_encoder(img, stereo=True)
        metas = dict(k2s_sensor=k2s_sensor,
                     intrins=intrin,
                     post_rots=post_rot,
                     post_trans=post_tran,
                     frustum=self.depth_net.cv_frustum.to(x),
                     cv_downsample=4,
                     downsample=self.depth_net.downsample,
                     grid_config=self.depth_net.grid_config,
                     cv_feat_list=[feat_prev_iv, stereo_feat])
        depth, context = self.depth_net(x, mlp_input, metas)
        bev_feat = self.img_view_transformer(
            [context, sensor2keyego, ego2global, intrin, post_rot, post_tran, bda,
             depth])
        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]
        return bev_feat, depth, stereo_feat
    
    def extract_img_feat(self, img, img_metas, **kwargs):
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
        bda, curr2adjsensor = img

        """ Extract features of images. """
        bev_feat_list = []
        depth_key_frame = None
        feat_prev_iv = None
        for fid in range(self.num_frame - 1, -1, -1):
            img, sensor2keyego, ego2global, intrin, post_rot, post_tran = \
                imgs[fid], sensor2keyegos[fid], ego2globals[fid], intrins[fid], \
                post_rots[fid], post_trans[fid]
            key_frame = fid == 0
            extra_ref_frame = fid == self.num_frame - self.extra_ref_frames
            if key_frame or self.with_prev:
                if self.align_after_view_transformation:
                    sensor2keyego, ego2global = sensor2keyegos[0], ego2globals[0]
                mlp_input = self.depth_net.get_mlp_input(
                    sensor2keyegos[0], ego2globals[0], intrin, post_rot, post_tran, bda)
                inputs_curr = (img, sensor2keyego, ego2global, intrin,
                               post_rot, post_tran, bda, mlp_input,
                               feat_prev_iv, curr2adjsensor[fid],
                               extra_ref_frame)
                if key_frame:
                    bev_feat, depth, feat_curr_iv = self.prepare_bev_feat(*inputs_curr)
                    depth_key_frame = depth
                else:
                    with torch.no_grad():
                        bev_feat, depth, feat_curr_iv = \
                            self.prepare_bev_feat(*inputs_curr)
                if not extra_ref_frame:
                    bev_feat_list.append(bev_feat)
                feat_prev_iv = feat_curr_iv
        
        if not self.with_prev:
            bev_feat_key = bev_feat_list[0]
            if len(bev_feat_key.shape) ==4:
                b,c,h,w = bev_feat_key.shape
                bev_feat_list = \
                    [torch.zeros([b,
                                  c * (self.num_frame -
                                       self.extra_ref_frames - 1),
                                  h, w]).to(bev_feat_key), bev_feat_key]
            else:
                b, c, z, h, w = bev_feat_key.shape
                bev_feat_list = \
                    [torch.zeros([b,
                                  c * (self.num_frame -
                                       self.extra_ref_frames - 1), z,
                                  h, w]).to(bev_feat_key), bev_feat_key]
        
        bev_feat = torch.cat(bev_feat_list, dim=1)
        x = self.bev_encoder(bev_feat)
        return x, depth_key_frame
    
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
        img_inputs = self.prepare_inputs(img_inputs, stereo=True)

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

        img_inputs = self.prepare_inputs(img_inputs, stereo=True)
        
        x, depth = self.extract_img_feat(img_inputs, img_metas, **kwargs)
        
        outs = self.occ_head(x)
        occ_pred = self.occ_head.get_occ_pred(outs)

        results_dict = dict()
        results_dict['occ_field'] = outs['occ_field']
        results_dict['language_field'] = outs['language_field']
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