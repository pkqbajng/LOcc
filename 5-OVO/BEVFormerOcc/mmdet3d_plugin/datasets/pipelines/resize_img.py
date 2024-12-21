from distutils.util import copydir_run_2to3
from ssl import PROTOCOL_TLS_SERVER
from unittest import result
from PIL import Image
from turtle import pos
import mmcv
import numpy as np
from mmdet.datasets.builder import PIPELINES
import torch

@PIPELINES.register_module()
class ResizeImages(object):
    def __init__(self, img_size=(256, 704)):
        self.img_size = img_size
    
    def sample_augmentation(self, H, W):
        fH, fW = self.img_size
        resize = float(fW) / float(W)
        resize_dims = (int(W*resize), int(H*resize))
        newW, newH = resize_dims
        crop_h = newH - fH
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        return resize, resize_dims, crop
    
    def img_transform_core(self, img, resize_dims, crop):
        img = img.resize(resize_dims)
        img = img.crop(crop)

        return img
    
    def img_transform(self, img, post_rot, post_tran, resize, resize_dims, crop):
        img = self.img_transform_core(img, resize_dims, crop)

        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        A = self.get_rot()
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
        return img, post_rot, post_tran
    
    def get_rot(self):
        return torch.Tensor([
            [1, 0],
            [0, 1],
        ])
    
    def __call__(self, results):
        img_inputs = []
        imgs = results['img']
        H, W = imgs[0].shape[:-1]

        resize, resize_dims, crop = self.sample_augmentation(H, W)

        for img in imgs:
            image = Image.fromarray(np.uint8(img))
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            image, post_rot2, post_tran2 = self.img_transform(image, post_rot, post_tran, 
                resize=resize, resize_dims=resize_dims, crop=crop)

            img_inputs.append(np.array(image))
        
        img_inputs = np.stack(img_inputs, axis=-1)
        img_inputs = img_inputs.astype(np.float32)
        
        results['img'] = [img_inputs[..., i] for i in range(img_inputs.shape[-1])]
        results['img_shape'] = img_inputs.shape
        results['pad_shape'] = img_inputs.shape
        results['post_rot'] = post_rot2
        results['post_tran'] = post_tran2

        return results