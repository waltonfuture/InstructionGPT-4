import random
import json
import torch
import os
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
import argparse
from transformers import AutoTokenizer, LlamaModel
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

def parse_args():
    parser = argparse.ArgumentParser(description="Ablation")
    # clip llama2
    parser.add_argument("--text_encoder", type=str, default="llama2")
    args = parser.parse_args()
    return args

tokenizer = AutoTokenizer.from_pretrained("daryl149/llama-2-7b-chat-hf", use_fast=False)
languagemodel = LlamaModel.from_pretrained("daryl149/llama-2-7b-chat-hf").half().cuda()

def get_text_embed(text, type):
    if type == 'llama2':
        inputs = tokenizer(text, return_tensors="pt")
        outputs = languagemodel(inputs.input_ids.cuda(), return_dict = True)
        hidden_states = outputs[0]
        text_features = hidden_states[torch.arange(1), -1]
        return text_features
    if type == 'clip':
        pass
        

def main(args):

    pca = PCA(n_components=6)
 
    text_vectors = []

    with torch.no_grad():
   
        save_dir = f'/path/to/cc_sbu_align/'
        save_file = save_dir + 'filter_cap.json'
        with open(save_file, 'r', encoding='utf-8') as f:
            splitdata = json.load(f)
        anno = splitdata['annotations']

        text_vectors = []

        for i in range(len(anno)):
            image_id = anno[i]["image_id"]
            text = anno[i]["caption"]
            text_embed = get_text_embed(text, args.text_encoder)
            text_vectors.append(text_embed)

        text_matrix = torch.cat(text_vectors, dim=0)
        # dimension induction
        text_matrix = torch.Tensor(pca.fit_transform(text_matrix.cpu())) #!

    torch.save(text_matrix, 'dataset/all_text_matrix.pt')
    

if __name__ == "__main__":
    args = parse_args()
    main(args)