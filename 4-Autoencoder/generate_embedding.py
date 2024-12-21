import os
import json
import torch
import argparse
from PIL import Image
import open_clip
from tqdm import tqdm

query_texts = ['barrier', 'traffic barrier', 'bicycle', 'bus', 'car', 
                'vehicle', 'sedan', 'SUV', 'construction vehicle', 
                'crane', 'motorcycle', 'pedestrian', 'person', 
                'traffic cone', 'trailer', 'delivery trailer', 'truck', 
                'driveable surface', 'road', 'water', 'river', 
                'lake', 'sidewalk', 'terrain', 'grass', 
                'building', 'wall', 'traffic light','sign', 
                'parking meter', 'hydrant', 'fence', 'vegetation', 'tree', 'sky']

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/occ3d")
    parser.add_argument('--ovo_root', default="data/occ3d/san_gts_qwen_scene", type=str)
    parser.add_argument('--query', action='store_true')
    parser.add_argument('--query_embedding_file', default='data/occ3d/text_embedding/query.json', type=str)
    parser.add_argument('--low_dimension_query_embedding_file', default='data/occ3d/text_embedding/query_128.json', type=str)
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

def generate_embeddings(vocab):
    text_features = []

    for classname in vocab:
        texts = [prompt_templates.format(classname)]
        texts = tokenizer(texts).cuda()
        with torch.no_grad():
            text_feature = model.encode_text(texts)
        text_feature /= text_feature.norm(dim=-1, keepdim=True)

        text_features.append(text_feature.detach().squeeze(0).cpu().numpy())

    text_embeddings = {}
    for i in range(len(vocab)):
        text_embeddings[vocab[i]] = text_features[i].tolist()

    return text_embeddings

if __name__ == '__main__':
    args = parse_config()

    data_root = args.data_root
    gt_root = os.path.join(data_root, 'gts')
    
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
    model = model.eval().cuda()
    tokenizer = open_clip.get_tokenizer('ViT-B-16')

    prompt_templates = 'a photo of {}'

    if args.query:
        vocab = query_texts
        text_embeddings = generate_embeddings(vocab)
        json_dict = json.dumps(text_embeddings)
        assert args.low_dimension_query_embedding_file is not None
        with open(args.low_dimension_query_embedding_file, "w") as f:
            f.write(json_dict)
    else:
        for i in tqdm(range(args.start, args.end)):
            scene = str(i).zfill(4)
            scene_name = 'scene-{}'.format(scene)
            print(f"Processing {scene_name}")
            scene_dir = os.path.join(args.ovo_root, scene_name)

            if os.path.exists(scene_dir):
                vocab_root = os.path.join(data_root, 'qwen_texts', scene_name, 'scene_vocabulary.txt')
                vocab = load_txt(vocab_root)
                text_embeddings = generate_embeddings(vocab)
                json_dict = json.dumps(text_embeddings)
                with open(os.path.join(scene_dir, 'vocab.json'), "w") as f:
                    f.write(json_dict)