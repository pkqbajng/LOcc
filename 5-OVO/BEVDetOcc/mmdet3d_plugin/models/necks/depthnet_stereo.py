import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, force_fp32
from mmdet.models.backbones.resnet import BasicBlock
from mmcv.cnn import build_conv_layer
from torch.cuda.amp.autocast_mode import autocast
from torch.utils.checkpoint import checkpoint
from mmdet.models import NECKS


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()
        dilations = [1, 6, 12, 18]
        self.aspp1 = _ASPPModule(
            inplanes,
            mid_channels,
            1,
            padding=0,
            dilation=dilations[0],
            BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[1],
            dilation=dilations[1],
            BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[2],
            dilation=dilations[2],
            BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[3],
            dilation=dilations[3],
            BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(
            int(mid_channels * 5), inplanes, 1, bias=False)
        self.bn1 = BatchNorm(inplanes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        """
        Args:
            x: (B*N, C, fH, fW)
        Returns:
            x: (B*N, C, fH, fW)
        """
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(
            x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # (B*N, 5*C', fH, fW)

        x = self.conv1(x)   # (B*N, C, fH, fW)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        """
        Args:
            x: (B*N_views, 27)
        Returns:
            x: (B*N_views, C)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        """
        Args:
            x: (B*N_views, C_mid, fH, fW)
            x_se: (B*N_views, C_mid, 1, 1)
        Returns:
            x: (B*N_views, C_mid, fH, fW)
        """
        x_se = self.conv_reduce(x_se)     # (B*N_views, C_mid, 1, 1)
        x_se = self.act1(x_se)      # (B*N_views, C_mid, 1, 1)
        x_se = self.conv_expand(x_se)   # (B*N_views, C_mid, 1, 1)
        return x * self.gate(x_se)      # (B*N_views, C_mid, fH, fW)

@NECKS.register_module()
class DepthNetStereo(BaseModule):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 context_channels,
                 grid_config,
                 input_size=None,
                 downsample=16,
                 use_dcn=True,
                 use_aspp=True,
                 with_cp=False,
                 stereo=False,
                 sid=False,
                 bias=0.0,
                 cost_volume_strides=[2, 2],
                 aspp_mid_channels=-1,
                 loss_depth_weight=3.0,
                 **kwargs
                 ):
        super(DepthNetStereo, self).__init__()
        self.sid = sid
        self.grid_config = grid_config
        self.downsample = downsample
        self.loss_depth_weight = loss_depth_weight

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        # 生成context feature
        self.context_conv = nn.Conv2d(
            mid_channels, context_channels, kernel_size=1, stride=1, padding=0)

        self.bn = nn.BatchNorm1d(27)
        self.depth_mlp = Mlp(in_features=27, hidden_features=mid_channels, out_features=mid_channels)
        self.depth_se = SELayer(channels=mid_channels)  # NOTE: add camera-aware
        self.context_mlp = Mlp(in_features=27, hidden_features=mid_channels, out_features=mid_channels)
        self.context_se = SELayer(channels=mid_channels)  # NOTE: add camera-aware
        depth_conv_input_channels = mid_channels
        downsample = None

        self.cv_frustum = self.create_frustum(grid_config['depth'], input_size, downsample=4)
        depth_channels = self.D

        if stereo:
            depth_conv_input_channels += depth_channels
            downsample = nn.Conv2d(depth_conv_input_channels,
                                    mid_channels, 1, 1, 0)
            cost_volumn_net = []
            for stage in range(len(cost_volume_strides)):
                cost_volumn_net.extend([
                    nn.Conv2d(depth_channels, depth_channels, kernel_size=3,
                              stride=cost_volume_strides[stage], padding=1),
                    nn.BatchNorm2d(depth_channels)])
            self.cost_volumn_net = nn.Sequential(*cost_volumn_net)
            self.bias = bias

        # 3个残差blocks
        depth_conv_list = [BasicBlock(depth_conv_input_channels, mid_channels,
                                      downsample=downsample),
                           BasicBlock(mid_channels, mid_channels),
                           BasicBlock(mid_channels, mid_channels)]
        if use_aspp:
            if aspp_mid_channels < 0:
                aspp_mid_channels = mid_channels
            depth_conv_list.append(ASPP(mid_channels, aspp_mid_channels))
        if use_dcn:
            depth_conv_list.append(
                build_conv_layer(
                    cfg=dict(
                        type='DCN',
                        in_channels=mid_channels,
                        out_channels=mid_channels,
                        kernel_size=3,
                        padding=1,
                        groups=4,
                        im2col_step=128,
                    )))
        depth_conv_list.append(
            nn.Conv2d(
                mid_channels,
                depth_channels,
                kernel_size=1,
                stride=1,
                padding=0))
        self.depth_conv = nn.Sequential(*depth_conv_list)
        self.with_cp = with_cp
        self.depth_channels = depth_channels

    # ----------------------------------------- 用于建立cost volume ----------------------------------
    def gen_grid(self, metas, B, N, D, H, W, hi, wi):
        """
        Args:
            metas: dict{
                k2s_sensor: (B, N_views, 4, 4)
                intrins: (B, N_views, 3, 3)
                post_rots: (B, N_views, 3, 3)
                post_trans: (B, N_views, 3)
                frustum: (D, fH_stereo, fW_stereo, 3)  3:(u, v, d)
                cv_downsample: 4,
                downsample: self.img_view_transformer.downsample=16,
                grid_config: self.img_view_transformer.grid_config,
                cv_feat_list: [feat_prev_iv, stereo_feat]
            }
            B: batchsize
            N: N_views
            D: D
            H: fH_stereo
            W: fW_stereo
            hi: H_img
            wi: W_img
        Returns:
            grid: (B*N_views, D*fH_stereo, fW_stereo, 2)
        """
        frustum = metas['frustum']      # (D, fH_stereo, fW_stereo, 3)  3:(u, v, d)
        # 逆图像增广:
        points = frustum - metas['post_trans'].view(B, N, 1, 1, 1, 3)
        points = torch.inverse(metas['post_rots']).view(B, N, 1, 1, 1, 3, 3) \
            .matmul(points.unsqueeze(-1))   # (B, N_views, D, fH_stereo, fW_stereo, 3, 1)

        # (u, v, d) --> (du, dv, d)
        # (B, N_views, D, fH_stereo, fW_stereo, 3, 1)
        points = torch.cat(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)

        # cur_pixel --> curr_camera --> prev_camera
        rots = metas['k2s_sensor'][:, :, :3, :3].contiguous()
        trans = metas['k2s_sensor'][:, :, :3, 3].contiguous()
        combine = rots.matmul(torch.inverse(metas['intrins']))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points)
        points += trans.view(B, N, 1, 1, 1, 3, 1)   # (B, N_views, D, fH_stereo, fW_stereo, 3, 1)

        neg_mask = points[..., 2, 0] < 1e-3
        # prev_camera --> prev_pixel
        points = metas['intrins'].view(B, N, 1, 1, 1, 3, 3).matmul(points)
        # (du, dv, d) --> (u, v)   (B, N_views, D, fH_stereo, fW_stereo, 2, 1)
        points = points[..., :2, :] / points[..., 2:3, :]

        # 图像增广
        points = metas['post_rots'][..., :2, :2].view(B, N, 1, 1, 1, 2, 2).matmul(
            points).squeeze(-1)
        points += metas['post_trans'][..., :2].view(B, N, 1, 1, 1, 2)   # (B, N_views, D, fH_stereo, fW_stereo, 2)

        px = points[..., 0] / (wi - 1.0) * 2.0 - 1.0
        py = points[..., 1] / (hi - 1.0) * 2.0 - 1.0
        px[neg_mask] = -2
        py[neg_mask] = -2
        grid = torch.stack([px, py], dim=-1)    # (B, N_views, D, fH_stereo, fW_stereo, 2)
        grid = grid.view(B * N, D * H, W, 2)    # (B*N_views, D*fH_stereo, fW_stereo, 2)
        return grid

    def calculate_cost_volumn(self, metas):
        """
        Args:
            metas: dict{
                k2s_sensor: (B, N_views, 4, 4)
                intrins: (B, N_views, 3, 3)
                post_rots: (B, N_views, 3, 3)
                post_trans: (B, N_views, 3)
                frustum: (D, fH_stereo, fW_stereo, 3)  3:(u, v, d)
                cv_downsample: 4,
                downsample: self.img_view_transformer.downsample=16,
                grid_config: self.img_view_transformer.grid_config,
                cv_feat_list: [feat_prev_iv, stereo_feat]
            }
        Returns:
            cost_volumn: (B*N_views, D, fH_stereo, fW_stereo)
        """
        prev, curr = metas['cv_feat_list']    # (B*N_views, C_stereo, fH_stereo, fW_stereo)
        group_size = 4
        _, c, hf, wf = curr.shape   #
        hi, wi = hf * 4, wf * 4     # H_img, W_img
        B, N, _ = metas['post_trans'].shape
        D, H, W, _ = metas['frustum'].shape
        grid = self.gen_grid(metas, B, N, D, H, W, hi, wi).to(curr.dtype)   # (B*N_views, D*fH_stereo, fW_stereo, 2)

        prev = prev.view(B * N, -1, H, W)   # (B*N_views, C_stereo, fH_stereo, fW_stereo)
        curr = curr.view(B * N, -1, H, W)   # (B*N_views, C_stereo, fH_stereo, fW_stereo)
        cost_volumn = 0
        # process in group wise to save memory
        for fid in range(curr.shape[1] // group_size):
            # (B*N_views, group_size, fH_stereo, fW_stereo)
            prev_curr = prev[:, fid * group_size:(fid + 1) * group_size, ...]
            wrap_prev = F.grid_sample(prev_curr, grid,
                                      align_corners=True,
                                      padding_mode='zeros')     # (B*N_views, group_size, D*fH_stereo, fW_stereo)
            # (B*N_views, group_size, fH_stereo, fW_stereo)
            curr_tmp = curr[:, fid * group_size:(fid + 1) * group_size, ...]
            # (B*N_views, group_size, 1, fH_stereo, fW_stereo) - (B*N_views, group_size, D, fH_stereo, fW_stereo)
            # --> (B*N_views, group_size, D, fH_stereo, fW_stereo)
            # https://github.com/HuangJunJie2017/BEVDet/issues/278
            cost_volumn_tmp = curr_tmp.unsqueeze(2) - \
                              wrap_prev.view(B * N, -1, D, H, W)
            cost_volumn_tmp = cost_volumn_tmp.abs().sum(dim=1)      # (B*N_views, D, fH_stereo, fW_stereo)
            cost_volumn += cost_volumn_tmp  # (B*N_views, D, fH_stereo, fW_stereo)
        if not self.bias == 0:
            invalid = wrap_prev[:, 0, ...].view(B * N, D, H, W) == 0
            cost_volumn[invalid] = cost_volumn[invalid] + self.bias

        # matching cost --> prob
        cost_volumn = - cost_volumn
        cost_volumn = cost_volumn.softmax(dim=1)
        return cost_volumn
    # ----------------------------------------- 用于建立cost volume --------------------------------------

    def forward(self, x, mlp_input, stereo_metas=None):
        """
        Args:
            x: (B, N_views, C, fH, fW)
            mlp_input: (B, N_views, 27)
            stereo_metas:  None or dict{
                k2s_sensor: (B, N_views, 4, 4)
                intrins: (B, N_views, 3, 3)
                post_rots: (B, N_views, 3, 3)
                post_trans: (B, N_views, 3)
                frustum: (D, fH_stereo, fW_stereo, 3)  3:(u, v, d)
                cv_downsample: 4,
                downsample: self.img_view_transformer.downsample=16,
                grid_config: self.img_view_transformer.grid_config,
                cv_feat_list: [feat_prev_iv, stereo_feat]
            }
        Returns:
            x: (B*N_views, D+C_context, fH, fW)
        """
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))     # (B*N_views, 27)
        x = self.reduce_conv(x)     # (B*N_views, C_mid, fH, fW)

        # (B*N_views, 27) --> (B*N_views, C_mid) --> (B*N_views, C_mid, 1, 1)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)    # (B*N_views, C_mid, fH, fW)
        context = self.context_conv(context)        # (B*N_views, C_context, fH, fW)
        context = context.view(B, N, -1, H, W)

        # (B*N_views, 27) --> (B*N_views, C_mid) --> (B*N_views, C_mid, 1, 1)
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)      # (B*N_views, C_mid, fH, fW)

        if not stereo_metas is None:
            if stereo_metas['cv_feat_list'][0] is None:
                BN, _, H, W = x.shape
                scale_factor = float(stereo_metas['downsample'])/\
                               stereo_metas['cv_downsample']
                cost_volumn = \
                    torch.zeros((BN, self.depth_channels,
                                 int(H*scale_factor),
                                 int(W*scale_factor))).to(x)
            else:
                with torch.no_grad():
                    # https://github.com/HuangJunJie2017/BEVDet/issues/278
                    cost_volumn = self.calculate_cost_volumn(stereo_metas)      # (B*N_views, D, fH_stereo, fW_stereo)
            cost_volumn = self.cost_volumn_net(cost_volumn)     # (B*N_views, D, fH, fW)
            depth = torch.cat([depth, cost_volumn], dim=1)      # (B*N_views, C_mid+D, fH, fW)
        if self.with_cp:
            depth = checkpoint(self.depth_conv, depth)
        else:
            # 3*res blocks +ASPP/DCN + Conv(c_mid-->D)
            depth = self.depth_conv(depth)  # x: (B*N_views, C_mid, fH, fW) --> (B*N_views, D, fH, fW)
        
        depth = depth.softmax(dim=1)
        return depth, context
    
    def get_mlp_input(self, sensor2ego, ego2global, intrin, post_rot, post_tran, bda):
        B, N, _, _ = sensor2ego.shape
        bda = bda.view(B, 1, 3, 3).repeat(1, N, 1, 1)
        mlp_input = torch.stack([
            intrin[:, :, 0, 0],
            intrin[:, :, 1, 1],
            intrin[:, :, 0, 2],
            intrin[:, :, 1, 2],
            post_rot[:, :, 0, 0],
            post_rot[:, :, 0, 1],
            post_tran[:, :, 0],
            post_rot[:, :, 1, 0],
            post_rot[:, :, 1, 1],
            post_tran[:, :, 1],
            bda[:, :, 0, 0],
            bda[:, :, 0, 1],
            bda[:, :, 1, 0],
            bda[:, :, 1, 1],
            bda[:, :, 2, 2],], dim=-1)
        sensor2ego = sensor2ego[:,:,:3,:].reshape(B, N, -1)
        mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)
        return mlp_input

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
    
    def create_frustum(self, depth_cfg, input_size, downsample):
        """Generate the frustum template for each image.

        Args:
            depth_cfg (tuple(float)): Config of grid alone depth axis in format
                of (lower_bound, upper_bound, interval).
            input_size (tuple(int)): Size of input images in format of (height,
                width).
            downsample (int): Down sample scale factor from the input size to
                the feature size.
        Returns:
            frustum: (D, fH, fW, 3)  3:(u, v, d)
        """
        H_in, W_in = input_size
        H_feat, W_feat = H_in // downsample, W_in // downsample
        d = torch.arange(*depth_cfg, dtype=torch.float)\
            .view(-1, 1, 1).expand(-1, H_feat, W_feat)      # (D, fH, fW)
        self.D = d.shape[0]
        if self.sid:
            d_sid = torch.arange(self.D).float()
            depth_cfg_t = torch.tensor(depth_cfg).float()
            d_sid = torch.exp(torch.log(depth_cfg_t[0]) + d_sid / (self.D-1) *
                              torch.log((depth_cfg_t[1]-1) / depth_cfg_t[0]))
            d = d_sid.view(-1, 1, 1).expand(-1, H_feat, W_feat)

        x = torch.linspace(0, W_in - 1, W_feat,  dtype=torch.float)\
            .view(1, 1, W_feat).expand(self.D, H_feat, W_feat)      # (D, fH, fW)
        y = torch.linspace(0, H_in - 1, H_feat,  dtype=torch.float)\
            .view(1, H_feat, 1).expand(self.D, H_feat, W_feat)      # (D, fH, fW)

        return torch.stack((x, y, d), -1)    # (D, fH, fW, 3)  3:(u, v, d)


class DepthAggregation(nn.Module):
    """pixel cloud feature extraction."""

    def __init__(self, in_channels, mid_channels, out_channels):
        super(DepthAggregation, self).__init__()

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True),
            # nn.BatchNorm3d(out_channels),
            # nn.ReLU(inplace=True),
        )

    @autocast(False)
    def forward(self, x):
        x = checkpoint(self.reduce_conv, x)
        short_cut = x
        x = checkpoint(self.conv, x)
        x = short_cut + x
        x = self.out_conv(x)
        return x