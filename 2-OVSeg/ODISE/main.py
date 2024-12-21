import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from odise.config import instantiate_odise
from detectron2.data import MetadataCatalog
from odise.checkpoint import ODISECheckpointer
from detectron2.utils.env import seed_all_rng
from detectron2.config import LazyConfig, instantiate
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.utils.visualizer import random_color
from odise.engine.defaults import get_model_from_module
from nuscenes.nuscenes import NuScenes

def parse_config():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    
    parser.add_argument(
        "--config-file",
        default="configs/Panoptic/odise_label_coco_50e.py",
        type=str,
        help="path to config file",
    )
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = LazyConfig.load(args.config_file)
    cfg.model.overlap_threshold = 0
    cfg.model.clip_head.alpha = 0.35
    cfg.model.clip_head.beta = 0.65
    seed_all_rng(42)

    dataset_cfg = cfg.dataloader.test
    wrapper_cfg = cfg.dataloader.wrapper

    camera_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_FRONT_RIGHT']
    
    extra_classes = []
    vocab = "black pickup truck, pickup truck; blue sky, sky"
    for words in vocab.split(";"):
        extra_classes.append([word.strip() for word in words.split(",")])
    
    extra_colors = [random_color(rgb=True, maximum=1) for _ in range(len(extra_classes))]
    demo_thing_classes = extra_classes
    demo_stuff_classes = []
    demo_thing_colors = extra_colors
    demo_stuff_colors = []

    demo_metadata = MetadataCatalog.get("odise_demo_metadata")
    demo_metadata.thing_classes = [c[0] for c in demo_thing_classes]
    demo_metadata.stuff_classes = [
        *demo_metadata.thing_classes,
        *[c[0] for c in demo_stuff_classes],
    ]
    demo_metadata.thing_colors = demo_thing_colors
    demo_metadata.stuff_colors = demo_thing_colors + demo_stuff_colors
    demo_metadata.stuff_dataset_id_to_contiguous_id = {
        idx: idx for idx in range(len(demo_metadata.stuff_classes))
    }
    demo_metadata.thing_dataset_id_to_contiguous_id = {
        idx: idx for idx in range(len(demo_metadata.thing_classes))
    }

    wrapper_cfg.labels = demo_thing_classes + demo_stuff_classes
    wrapper_cfg.metadata = demo_metadata

    model = instantiate_odise(cfg.model)
    model.to(cfg.train.device)
    ODISECheckpointer(model).load("checkpoints/odise_label_coco_50e-b67d2efc.pth")

    while "model" in wrapper_cfg:
        wrapper_cfg = wrapper_cfg.model
    
    wrapper_cfg.model = get_model_from_module(model)
    
    inference_model = instantiate(cfg.dataloader.wrapper)
    inference_model = inference_model.eval()
    aug = instantiate(dataset_cfg.mapper).augmentations

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
                    img = utils.read_image(img_path, format="RGB")
                    height, width = img.shape[:2]

                    aug_input = T.AugInput(img, sem_seg=None)
                    aug(aug_input)
                    image = aug_input.image
                    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                    inputs = {"image": image, "height": height, "width": width}
                    texts = [[item] for item in vocabulary]
                    with torch.no_grad():
                        predictions = inference_model([inputs], vocabulary=texts)[0]
                    seg_result = predictions['sem_seg']

                    seg_result = predictions['sem_seg'].argmax(0)
                    seg_result = seg_result.cpu().numpy().astype(np.uint8)
                    
                    save_path = os.path.join(output_root, filename.replace('.jpg', '.png'))
                    cv2.imwrite(save_path, seg_result)