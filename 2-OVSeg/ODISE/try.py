from ast import arg
from hashlib import md5
import os
from statistics import mode
from symbol import import_as_name

import torch
import argparse


from odise.config import instantiate_odise
from detectron2.data import MetadataCatalog
from odise.checkpoint import ODISECheckpointer
from detectron2.utils.env import seed_all_rng
from detectron2.config import LazyConfig, instantiate
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.utils.visualizer import random_color
from odise.engine.defaults import get_model_from_module

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        default="configs/Panoptic/odise_label_coco_50e.py",
        type=str,
        help="path to config file",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = LazyConfig.load(args.config_file)
    cfg.model.overlap_threshold = 0
    cfg.model.clip_head.alpha = 0.35
    cfg.model.clip_head.beta = 0.65
    seed_all_rng(42)

    dataset_cfg = cfg.dataloader.test
    wrapper_cfg = cfg.dataloader.wrapper

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

    img = utils.read_image('demo/examples/coco.jpg', format="RGB")
    height, width = img.shape[:2]
    aug_input = T.AugInput(img, sem_seg=None)
    aug(aug_input)
    image = aug_input.image
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    inputs = {"image": image, "height": height, "width": width}
    predictions = inference_model([inputs])[0]
    print("test")