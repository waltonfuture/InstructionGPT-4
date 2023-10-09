import random
import json
import torch
from PIL import Image
import open_clip
import os
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

def parse_args():
    parser = argparse.ArgumentParser(description="Ablation")
    parser.add_argument("--text_encoder", type=str, default="llama2")
    args = parser.parse_args()
    return args

Atensor = torch.tensor(0).cuda()
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k')
clip_model = clip_model.cuda()

def get_image_embed(image):
    image = preprocess(image).unsqueeze(0).cuda()
    image_features = clip_model.encode_image(image)
    return image_features


def main(args):
    
    pca = PCA(n_components=6)

    image_vectors = [] 

    image_matrixes = [] 
    
    with torch.no_grad():
        for i in range(30):
            save_dir = f'/path/to/test_dataset/split/{i}/cc_sbu_align/'

            save_file = save_dir + 'filter_cap.json'
            with open(save_file, 'r', encoding='utf-8') as f:
                splitdata = json.load(f)
            splitanno = splitdata['annotations']

            image_vectors = [] 

            for i in range(len(splitanno)):
                image_id = splitanno[i]["image_id"]
                img = Image.open(f"/path/to/datasets/cc_sbu_align/image/{image_id}.jpg")
                
                img_embed = get_image_embed(img)
                image_vectors.append(img_embed)

            image_matrix = torch.cat(image_vectors, dim=0)
            # dimension induction
            image_matrix = torch.Tensor(pca.fit_transform(image_matrix.cpu())) #!

            image_matrixes.append(image_matrix)
            
    image_matrixes = torch.stack(image_matrixes)
    #pooled_image_matrix = torch.mean(image_matrixes, dim=1) #!
    print(image_matrixes.shape)
    print(image_matrix.shape)
    torch.save(image_matrixes, 'dataset/image_matrix.pt')

if __name__ == "__main__":
    args = parse_args()
    main(args)