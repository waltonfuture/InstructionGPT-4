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
    # clip llama2
    parser.add_argument("--text_encoder", type=str, default="llama2")
    args = parser.parse_args()
    return args

clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k')
clip_model = clip_model.cuda()

def get_image_embed(image):
    image = preprocess(image).unsqueeze(0).cuda()
    image_features = clip_model.encode_image(image)
    return image_features


def main(args):

    pca = PCA(n_components=6)

    image_vectors = [] 
 
    
    with torch.no_grad():
        save_dir = f'/path/to/cc_sbu_align/' # raw datasets' path
        save_file = save_dir + 'filter_cap.json'
        with open(save_file, 'r', encoding='utf-8') as f:
            splitdata = json.load(f)
        anno = splitdata['annotations']

        image_vectors = [] 

        for i in range(len(anno)):
            image_id = anno[i]["image_id"]
            img = Image.open(f"/path/to/cc_sbu_align/image/{image_id}.jpg")
            
            img_embed = get_image_embed(img)
            image_vectors.append(img_embed)

        image_matrix = torch.cat(image_vectors, dim=0)
        # dimension induction (if used in blip2, pca is not needed)
        image_matrix = torch.Tensor(pca.fit_transform(image_matrix.cpu())) #!


    print(image_matrix.shape) #[3439, 4]
    torch.save(image_matrix, 'dataset/all_image_matrix.pt')

if __name__ == "__main__":
    args = parse_args()
    main(args)