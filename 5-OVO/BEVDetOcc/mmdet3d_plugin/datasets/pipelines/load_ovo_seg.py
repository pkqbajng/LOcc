import torch
import numpy as np
from PIL import Image

from mmdet3d.datasets.builder import PIPELINES

@PIPELINES.register_module()
class LoadOVOSeg(object):
    def __init__(
        self,
        ovo_root='san_seg_vocab2',
        ignore_label=32
    ):
        self.ovo_root = ovo_root
        self.ignore_label = ignore_label
    
    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims, resample=Image.NEAREST)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate, resample=Image.NEAREST)
        return img
    
    def get_inputs(self, results):
        ovo_segs = []
        ovo_seg_masks = []
        img_augs_list = results['img_augs_list']
        cam_names = results['cam_names']
        for cam_name in cam_names:
            cam_data = results['curr']['cams'][cam_name]
            filename = cam_data['data_path']
            resize, resize_dims, crop, flip, rotate = img_augs_list[cam_name]

            ovo_seg_filename = filename.replace('samples', self.ovo_root + '/samples').replace('.jpg', '.png')
            ovo_seg = Image.open(ovo_seg_filename)
            ovo_seg = self.img_transform_core(ovo_seg, resize_dims, crop, flip, rotate)
            ovo_seg = torch.from_numpy(np.array(ovo_seg))
            ovo_seg_mask = (ovo_seg != 255) * (ovo_seg != self.ignore_label)
            ovo_segs.append(ovo_seg)
            ovo_seg_masks.append(ovo_seg_mask)

        ovo_segs = torch.stack(ovo_segs)
        ovo_seg_masks = torch.stack(ovo_seg_masks)
        return ovo_segs, ovo_seg_masks

    def __call__(self, results):
        ovo_segs, ovo_seg_masks = self.get_inputs(results)
        results['ovo_segs'] = ovo_segs
        results['ovo_seg_masks'] = ovo_seg_masks
        return results