import os
import torch
from mmdet3d.models import build_model
from .basemodel import LightningBaseModel
from .occ_metrics import Metric_SSC
from mmcv.runner.checkpoint import load_checkpoint

class pl_model(LightningBaseModel):
    def __init__(
        self,
        config
    ):
        super(pl_model, self).__init__(config)

        model_config = config['model']
        self.model = build_model(model_config)
        self.model.init_weights()
        if 'load_from' in config and config['load_from'] is not None:
            load_checkpoint(self.model, config['load_from'], map_location='cpu')
        
        self.num_cls = config['num_classes']
        self.class_names = config['occupancy_classes']

        self.train_metrics = Metric_SSC(num_classes=config['num_classes'], use_image_mask=True)
        self.val_metrics = Metric_SSC(config['num_classes'], use_image_mask=True)
        self.test_metrics = Metric_SSC(config['num_classes'], use_image_mask=True)
        self.config = config

    def forward(self, **kwargs):
        return self.model(**kwargs)
    
    def training_step(self, batch, batch_idx):
        output_dict = self.forward(**batch)

        loss = 0
        losses = output_dict['losses']

        for key in losses.keys():
            self.log("train/" + key,
                    losses[key].detach(),
                    on_epoch=True,
                    sync_dist=True)
            
            loss += losses[key]
        
       
        pred = output_dict['occ_pred'].detach().cpu().numpy()
        gt_occ = output_dict['voxel_semantics'].detach().cpu().numpy()
        mask = output_dict['mask_camera'].detach().cpu().numpy()
        
        self.train_metrics.add_batch(pred, gt_occ, mask_camera=mask)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        output_dict = self.forward(**batch)

        pred = output_dict['occ_pred'].detach().cpu().numpy()
        gt_occ = output_dict['voxel_semantics'].detach().cpu().numpy()
        mask = output_dict['mask_camera'].detach().cpu().numpy()

        self.val_metrics.add_batch(pred, gt_occ, mask_camera=mask)

    def validation_epoch_end(self, outputs):
        metric_list = [("train", self.train_metrics), ("val", self.val_metrics)]
        # metric_list = [("val", self.val_metrics)]
            
        metrics_list = metric_list
        for prefix, metric in metrics_list:
            stats = metric.count_miou()

            self.log("{}/mIoU".format(prefix), torch.tensor(stats["miou"], dtype=torch.float32), sync_dist=True)
            self.log("{}/IoU".format(prefix), torch.tensor(stats["iou"], dtype=torch.float32), sync_dist=True)
            self.log("{}/Precision".format(prefix), torch.tensor(stats["precision"], dtype=torch.float32), sync_dist=True)
            self.log("{}/Recall".format(prefix), torch.tensor(stats["recall"], dtype=torch.float32), sync_dist=True)
            metric.reset()
    
    def test_step(self, batch, batch_idx):
        output_dict = self.forward(**batch)
        pred = output_dict['occ_pred'].detach().cpu().numpy()
        gt_occ = output_dict['voxel_semantics'].detach().cpu().numpy()
        mask = output_dict['mask_camera'].detach().cpu().numpy()
        
        if gt_occ is not None:
            self.test_metrics.add_batch(pred, gt_occ, mask_camera=mask)
        else:
            gt_occ = None

        if self.config['visualize'] is not None:
            visualize_root = self.config['visualize']
            print(output_dict['img_metas'][0]['scene_idx'])
            scene = output_dict['img_metas'][0]['scene_idx']
            scene = str(scene).zfill(4)
            scene_name = 'scene-{}'.format(scene)
            token = output_dict['img_metas'][0]['sample_idx']
            dst_root = os.path.join(visualize_root, scene_name, token)
            os.makedirs(dst_root, exist_ok=True)
            import numpy as np
            np.save(os.path.join(dst_root, 'labes.npz'), pred)
        
    def test_epoch_end(self, outputs):
        metric_list = [("test", self.test_metrics)]
        # metric_list = [("val", self.val_metrics)]
        metrics_list = metric_list
        for prefix, metric in metrics_list:
            stats = metric.count_miou()

            self.log("{}/mIoU".format(prefix), torch.tensor(stats["miou"], dtype=torch.float32), sync_dist=True)
            self.log("{}/IoU".format(prefix), torch.tensor(stats["iou"], dtype=torch.float32), sync_dist=True)
            self.log("{}/Precision".format(prefix), torch.tensor(stats["precision"], dtype=torch.float32), sync_dist=True)
            self.log("{}/Recall".format(prefix), torch.tensor(stats["recall"], dtype=torch.float32), sync_dist=True)
            metric.reset()
        print(stats['iou_ssc'])