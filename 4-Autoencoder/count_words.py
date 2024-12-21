import os
import json
import torch
import pickle
import argparse
import numpy as np
from PIL import Image
import open_clip
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/occ3d")
    parser.add_argument("--ovo_root", default="data/occ3d/san_gts_qwen_scene", type=str)
    parser.add_argument("--embedding_file", type=str, default="data/occ3d/text_embedding/overall_embedding.json")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=1111)
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

    nusc = NuScenes(version='v1.0-trainval',
                    dataroot=data_root,
                    verbose=True)
    max_length = 10
    vocab_all = []
    vocab_x = []
    for i in tqdm(range(args.start, args.end)):
        scene = str(i).zfill(4)
        scene_name = 'scene-{}'.format(scene)
        print(f"Processing {scene_name}")

        scene_dir = os.path.join(args.ovo_root, scene_name)
        if os.path.exists(scene_dir):
            tmp_files = os.listdir(scene_dir)

            gt_files = []
            for tmp_file in tmp_files:
                if not tmp_file.endswith('.json'):
                    gt_files.append(tmp_file)
            
            valid_indexes = np.array([])
            for gt_file in gt_files:
                ovo_file = os.path.join(scene_dir, gt_file,  "label_ovo.pkl")
                vocab_root = os.path.join(data_root, 'qwen_texts_step3', scene_name, 'scene_vocabulary.txt')
                vocab = load_txt(vocab_root)
                with open(ovo_file, "rb") as f:
                    infos = pickle.load(f)

                ovo_gt = infos['ovo_gt']
                ovo_mask = (ovo_gt != 255) * (ovo_gt != len(vocab))
                indexes = np.unique(ovo_gt[ovo_mask])
                valid_indexes = np.union1d(valid_indexes, indexes)

        
            vocab_root = os.path.join(data_root, 'qwen_texts_step3', scene_name, 'scene_vocabulary.txt')
            vocab = load_txt(vocab_root)
            max_length = len(vocab) if max_length < len(vocab) else max_length
            valid_vocab = []
            for valid_index in valid_indexes:
                valid_vocab.append(vocab[int(valid_index)])
            vocab_all += valid_vocab
            vocab_x += vocab

    vocab_all = list(set(vocab_all))
    vocab_all.sort()
    vocab_x = list(set(vocab_x))
    
    print("Length of total vocabulary: ", len(vocab_all))
    text_embeddings = generate_embeddings(vocab_all)

    json_dict = json.dumps(text_embeddings)
    with open("overall_embedding.json", "w") as f:
        f.write(json_dict)