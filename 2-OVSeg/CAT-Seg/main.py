import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config

from cat_seg import add_cat_seg_config
from detectron2.engine.defaults import DefaultPredictor
from nuscenes.nuscenes import NuScenes

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_cat_seg_config(cfg)
    cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    if torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cuda"
    cfg.freeze()
    return cfg

templates = ['A photo of a {} in the scene',]

def parse_config():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    
    parser.add_argument("--config-file", default="configs/demo.yaml")
    parser.add_argument("--vocab_root", default="data/occ3d/qwen_texts_step3")
    parser.add_argument("--data_root", default="data/occ3d")
    parser.add_argument("--output_root", default="data/occ3d/cat_seg_qwen_scene")
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
    camera_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_FRONT_RIGHT']

    cfg = setup_cfg(args)

    predictor = DefaultPredictor(cfg)
    pred = predictor.model.sem_seg_head.predictor
    
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
            print(vocabulary)
                
            pred.test_class_texts = vocabulary
            pred.text_features_test = pred.class_embeddings(pred.test_class_texts, 
            ['A photo of a {} in the scene',],
            pred.clip_model).permute(1, 0, 2).float().repeat(1, 80, 1)
            
            index_list = os.listdir(scene_dir)
            for index in tqdm(index_list):
                rec = nusc.get('sample', index)
                
                for cam_name in camera_names:
                    cam_sample = nusc.get('sample_data', rec['data'][cam_name])
                    filename = cam_sample['filename']
                    os.makedirs(os.path.join(output_root, 'samples', cam_name), exist_ok=True)

                    img_path = os.path.join(data_root, filename)
                    img = cv2.imread(img_path)
                    predictions = predictor(img)
                    seg_result = predictions['sem_seg'].argmax(0)
                    seg_result = seg_result.cpu().numpy().astype(np.uint8)
                    
                    save_path = os.path.join(output_root, filename.replace('.jpg', '.png'))
                    cv2.imwrite(save_path, seg_result)