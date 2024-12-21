import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from predict import Predictor
from nuscenes.nuscenes import NuScenes

model_cfg = {
    "san_vit_b_16": {
        "config_file": "configs/san_clip_vit_res4_coco.yaml",
        "model_path": "./ckpts/san_vit_b_16.pth",
    },
    "san_vit_large_16": {
        "config_file": "configs/san_clip_vit_large_res4_coco.yaml",
        "model_path": "./ckpts/san_vit_large_14.pth",
    },
}

model_name = "san_vit_b_16"

def parse_config():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    
    parser.add_argument("--vocab_root", default="data/occ3d/qwen_texts")
    parser.add_argument("--data_root", default="data/occ3d")
    parser.add_argument("--output_root", default="data/occ3d/san_feat")
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--end', type=int, default=750)

    args = parser.parse_args()

    return args

def load_txt(file_path):
    words = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            for word in line.strip().split(';'):
                words.append(word)
    words = list(words)
    return words

if __name__ == '__main__':
    args = parse_config()

    data_root = args.data_root
    gt_root = os.path.join(data_root, 'gts')
    output_root = args.output_root
    predictor = Predictor(**model_cfg[model_name])
    camera_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_FRONT_RIGHT']
   
    nusc = NuScenes(version='v1.0-trainval',
                    dataroot=data_root,
                    verbose=True)
    print(f"start: {args.start}, {args.end}!")
    for i in tqdm(range(args.start, args.end)):
        scene = str(i).zfill(4)
        scene_name = 'scene-{}'.format(scene)
        print(f"Processing {scene_name}")
        scene_dir = os.path.join(gt_root, scene_name)
        if os.path.exists(scene_dir):
            scene_vocab_path = os.path.join(args.vocab_root, scene_name, 'scene_vocabulary.txt')
            scene_vocab = load_txt(scene_vocab_path)
            scene_vocab.append('sky')
            vocabulary = scene_vocab

            index_list = os.listdir(scene_dir)
            for index in tqdm(index_list):
                rec = nusc.get('sample', index)
                
                for cam_name in camera_names:
                    cam_sample = nusc.get('sample_data', rec['data'][cam_name])
                    filename = cam_sample['filename']
                    os.makedirs(os.path.join(output_root, 'samples', cam_name), exist_ok=True)

                    img_path = os.path.join(data_root, filename)
                    result = predictor.predict(
                        img_path, vocabulary=vocabulary)
                    
                    seg_feat = result['seg_feat'].detach().cpu()
                    seg_feat = seg_feat.squeeze(0).permute(2, 0, 1)
                    seg_feat = seg_feat.to(torch.float16)

                    save_path = os.path.join(output_root, filename.replace('.jpg', '.pt'))
                    torch.save(seg_feat, save_path)