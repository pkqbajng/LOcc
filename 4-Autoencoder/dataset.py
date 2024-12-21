import os
import json
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class Autoencoder_dataset(Dataset):
    def __init__(self, text_embedding_file):
        with open(text_embedding_file, 'r') as f1:
            info = json.load(f1)
        k_word_tokens = []
        for k in info:
            k_word_tokens.append(torch.Tensor(info[k]).unsqueeze(0))
        k_word_tokens = torch.cat(k_word_tokens)
        self.data = k_word_tokens

    def __getitem__(self, index):
        data = self.data[index].detach().clone()
        return data

    def __len__(self):
        return self.data.shape[0]