import os
import json
import torch
import pickle
import numpy as np
from mmdet.datasets.builder import PIPELINES

def load_text_embedding(text_embedding_file):
    with open(text_embedding_file, 'r') as f1:
        info = json.load(f1)
    
    text_embedding = []
    for k in info:
        text_embedding.append(torch.Tensor(info[k]).unsqueeze(0))
    text_embedding = torch.cat(text_embedding)

    return text_embedding

@PIPELINES.register_module()
class LoadOVOGTFromFile(object):
    def __init__(
        self,
        ovo_gt_root,
        data_root=None,
        scene_specific=False,
        ignore_label=32,
        text_embedding_file='vocab_128.json'
    ):
        self.data_root = data_root
        self.ovo_gt_root = ovo_gt_root
        self.scene_specific = scene_specific
        self.ignore_label = ignore_label
        self.text_embedding_file = text_embedding_file
        
    def __call__(self, results):
        occ_gt_path = results['occ_gt_path']
        ovo_gt_path = occ_gt_path.replace('gts', self.ovo_gt_root).replace('labels.npz', 'label_ovo.pkl')
        ovo_gt_path = os.path.join(self.data_root, ovo_gt_path)

        with open(ovo_gt_path, "rb") as f:
            infos = pickle.load(f)
        ovo_gt = infos['ovo_gt']
        if not self.scene_specific:
            ovo_gt_mask = (ovo_gt != 255) * (ovo_gt != self.ignore_label)
        else:
            parts = ovo_gt_path.rsplit('/', 2)
            text_embedding_file = os.path.join(parts[-3], self.text_embedding_file)
            text_embedding = load_text_embedding(text_embedding_file)
            ignore_label = text_embedding.shape[0]
            ovo_gt_mask = (ovo_gt != 255) * (ovo_gt != ignore_label)
            results['text_embedding_file'] = text_embedding_file

        ovo_gt = torch.from_numpy(ovo_gt)
        ovo_gt_mask = torch.from_numpy(ovo_gt_mask)

        if results.get('flip_dx', False):
            ovo_gt = torch.flip(ovo_gt, [0])
            ovo_gt_mask = torch.flip(ovo_gt_mask, [0])
        
        if results.get('flip_dy', False):
            ovo_gt = torch.flip(ovo_gt, [1])
            ovo_gt_mask = torch.flip(ovo_gt_mask, [1])
        
        results['ovo_gt'] = ovo_gt
        results['ovo_gt_mask'] = ovo_gt_mask

        return results