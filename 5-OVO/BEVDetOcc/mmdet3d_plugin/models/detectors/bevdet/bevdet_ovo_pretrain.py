import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32, BaseModule

from mmdet3d.models import DETECTORS
from mmdet3d.models import builder

@DETECTORS.register_module()
class BEVDetOVOPretrain(BaseModule):
    def __init__(
        self,
        img_backbone,
        img_neck,
        depth_net,
        plugin_head,
        **kwargs
    ):
        super().__init__()

        self.img_backbone = builder.build_backbone(img_backbone)
        self.img_neck = builder.build_neck(img_neck)

        self.depth_net = builder.build_neck(depth_net)
        self.plugin_head = builder.build_head(plugin_head)

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
    
    def extract_img_feat(self, img_inputs, img_metas, **kwargs):
        imgs, sensor2ego, ego2global, intrin, post_rot, post_tran, bda = img_inputs
        x = self.image_encoder(imgs)

        # mlp_input = self.depth_net.get_mlp_input(sensor2ego, ego2global, intrin, post_rot, post_tran, bda)
        depth, context = self.depth_net(x)
        return context, depth

    def forward_train(
        self, 
        img_inputs=None, 
        img_metas=None, 
        **kwargs):

        gt_depth = kwargs['gt_depth']
        ovo_segs = kwargs['ovo_segs']
        ovo_seg_masks = kwargs['ovo_seg_masks']

        img_inputs = self.prepare_inputs(img_inputs)
        
        context, depth = self.extract_img_feat(img_inputs, img_metas, **kwargs)
        language_feat = self.plugin_head(context)

        losses = dict()
        loss_depth = self.depth_net.get_depth_loss(gt_depth, depth)
        loss_language = self.plugin_head.get_language_loss(ovo_segs, ovo_seg_masks, language_feat)
        losses['loss_depth'] = loss_depth
        losses['loss_language'] = loss_language

        results_dict = dict()
        results_dict['losses'] = losses

        return results_dict
    
    def forward_test(
        self, 
        img_inputs=None, 
        img_metas=None, 
        **kwargs):
        
        img_inputs = self.prepare_inputs(img_inputs)
        context, depth = self.extract_img_feat(img_inputs, img_metas, **kwargs)
        language_feat = self.plugin_head(context)

        results_dict = dict()
        return results_dict
    
    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
