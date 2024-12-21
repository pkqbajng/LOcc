import os
import json
import torch
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from model import Autoencoder

def load_text_embedding(text_embedding_file):
    with open(text_embedding_file, 'r') as f1:
        info = json.load(f1)

    classnames = list(info.keys())
    k_word_tokens = []
    for k in info:
        k_word_tokens.append(torch.Tensor(info[k]).unsqueeze(0))
    k_word_tokens = torch.cat(k_word_tokens)
    
    return k_word_tokens, classnames

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', action='store_true')
    parser.add_argument('--query_embedding_file', default='openclip512.json', type=str)
    parser.add_argument('--low_dimension_query_embedding_file', default=None, type=str)
    parser.add_argument('--log_dir', type=str, default='openclip_512_128')
    parser.add_argument('--ovo_root', default="data/occ3d/san_gts_qwen_scene", type=str)
    parser.add_argument('--low_dimension_file', type=str, default='openclip_test.json')
    parser.add_argument('--encoder_dims',
                    nargs = '+',
                    type=int,
                    default=[256, 128, 128],
                    )
    parser.add_argument('--decoder_dims',
                    nargs = '+',
                    type=int,
                    default=[128, 256, 256, 512],
                    )
    args = parser.parse_args()

    encoder_hidden_dims = args.encoder_dims
    decoder_hidden_dims = args.decoder_dims

    ckpt_path = f"ckpt/{args.log_dir}/best_ckpt.pth"
    
    checkpoint = torch.load(ckpt_path)

    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")

    model.load_state_dict(checkpoint)
    model.eval()

    if args.query:
        query_embedding, classnames = load_text_embedding(args.query_embedding_file)
        low_dimension_text_embeddings = {}
        for i in range(len(classnames)):
            temp_low_dimension_feature = model.encode(query_embedding[i].unsqueeze(0).to("cuda:0"))
            low_dimension_text_embeddings[classnames[i]] = temp_low_dimension_feature.squeeze(0).detach().cpu().numpy().tolist()
        json_dict = json.dumps(low_dimension_text_embeddings)
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
                scene_embedding, classnames = load_text_embedding(os.path.join(scene_dir, 'vocab.json'))
                low_dimension_text_embeddings = {}
                for i in range(len(classnames)):
                    temp_low_dimension_feature = model.encode(scene_embedding[i].unsqueeze(0).to("cuda:0"))
                    low_dimension_text_embeddings[classnames[i]] = temp_low_dimension_feature.squeeze(0).detach().cpu().numpy().tolist()
                json_dict = json.dumps(low_dimension_text_embeddings)

                with open(os.path.join(scene_dir, 'vocab_128.json'), "w") as f:
                    f.write(json_dict)