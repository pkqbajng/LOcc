import os
import json
import torch
import shutil
import argparse
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import Autoencoder_dataset
from model import Autoencoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_embedding_file', default='openclip512.json', type=str)
    parser.add_argument('--log_dir', type=str, default='openclip_512_128')
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
    train_dataset = Autoencoder_dataset(args.text_embedding_file)

    test_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=1,
        shuffle=False, 
        num_workers=16, 
        drop_last=False   
    )

    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")

    model.load_state_dict(checkpoint)
    model.eval()

    text_features = []
    for idx, feature in tqdm(enumerate(test_loader)):
        data = feature.to("cuda:0")
        with torch.no_grad():
            outputs = model.encode(data) 
            text_features.append(outputs.detach().squeeze(0).cpu().numpy())
    
    text_embedding_file = args.text_embedding_file
    with open(text_embedding_file, 'r') as f1:
        info = json.load(f1)
    
    classnames = list(info.keys())
    print(classnames)
    text_embeddings = {}
    for i in range(len(classnames)):
        text_embeddings[classnames[i]] = text_features[i].tolist()

    json_dict = json.dumps(text_embeddings)
    with open(args.filename, "w") as f:
        f.write(json_dict)
