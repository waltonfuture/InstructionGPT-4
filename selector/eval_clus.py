import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from model.models import Selector
from torch.utils.data import TensorDataset, DataLoader
import json
import os
import shutil
import numpy as np
import random
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Selector")
    parser.add_argument("--feat_size", type=int, default=6)
    parser.add_argument("--layer_num", type=int, default=2)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--concat", type=int, default=0)
    parser.add_argument("--nhead", type=int, default=1)
    parser.add_argument("--residual", type=int, default=0)
    parser.add_argument("--data_num", type=int, default=200)
    args = parser.parse_args()
    return args

def main(args):
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    features_size = {'ins_embed_size': 4, 'image_feat_size': args.feat_size, 'text_feat_size': args.feat_size}

    model = Selector(
            feat_size=features_size,
            layer_num = args.layer_num,
            dimens = args.dimension,
            concat = args.concat,
            nhead = args.nhead,
            residual = args.residual
            )

    save_path = f'ckpt/nhead{args.nhead}_res{args.residual}_LayerNum{args.layer_num}_featsize{args.feat_size}_epoch{args.epoch}'
 
    model.load_state_dict(torch.load(os.path.join(save_path, 'best_model.pth')))  

    with open('/path/to/cc_sbu_align/filter_cap.json', 'r', encoding='utf-8') as f:
        part = json.load(f)
    data = part['annotations']
    
    with open('dataset/scores.json', 'r', encoding='utf-8') as f:
        final = json.load(f)
    
    clip_scores = torch.tensor(final["clip"])
    reward_scores = 10 * torch.tensor(final["reward"])
    length_scores = 10 * torch.tensor(final["length"])
    gpt_scores = torch.tensor(final["gpt"]) 

    X = torch.stack((clip_scores, reward_scores, length_scores, gpt_scores), dim=1)   
    X = X.unsqueeze(1)
    X = X.repeat(1, 114, 1)
    image_matrix = torch.load(f'dataset/all_image_matrix.pt')
    text_matrix = torch.load(f'dataset/all_text_matrix.pt')
    image_matrix = image_matrix.unsqueeze(1)
    image_matrix = image_matrix.repeat(1,114,1)
    text_matrix = text_matrix.unsqueeze(1)
    text_matrix = text_matrix.repeat(1,114,1)

    # total features: 
    model.eval()
    with torch.no_grad():
        scores = model(instruction_logits=X, image_feats=image_matrix, text_feats=text_matrix)

    # choose the top 200 data
    score = torch.mean(scores['scores'], dim=1)
    new = []

    with open('/path/to/cc_sbu_align/filter_cap.json', 'r', encoding='utf-8') as f:
        data_origin = json.load(f)
    new_annotations = []
    origin_annotations = data_origin['annotations']
    for i in range(len(origin_annotations)):
        image_id = origin_annotations[i]["image_id"]
        caption = origin_annotations[i]["caption"]
        new_annotation = {"image_id": image_id, "caption": caption}
        new_annotation["final_score"] = score[i] 
        # new_annotation["nli_score"] = get_nil_score(question, caption)
        new_annotations.append(new_annotation)
    with open('/path/to/cluster/spectral/clustering_results.json', 'r', encoding='utf-8') as f:
        cluster_data = json.load(f)
    cluster_annot = []
    score_dic = {annotation['image_id']: annotation['final_score'] for annotation in new_annotations}
    image_annotations = {annotation['image_id']: annotation['caption'] for annotation in origin_annotations}
    for folder_name, image_names in cluster_data.items():
        temp = []
        #print(folder_name)
        for image_name in image_names:
            caption = image_annotations.get(image_name)
            ql_score = score_dic.get(image_name)
            if caption is not None:
                new_temp = {"image_id": image_name, "caption": caption}  

                new_temp['final_score'] = ql_score                
                temp.append(new_temp) 
            else:
                print("error!")
                print(image_name)
        cluster_annot.append(temp)
    
    
    selected_annotations = []
    
    for i in cluster_annot:
        sorted_annotations = sorted(i, key=lambda x: int(x["final_score"]), reverse=True)
        num = int(204 * len(sorted_annotations) / 3439)
        if num==0:
            num = 1
        selected_annotations += sorted_annotations[:num]
    final = sorted(selected_annotations, key=lambda x: int(x['final_score']), reverse=True)
    print(len(final))
    final = final[:args.data_num]
    final = sorted(final, key=lambda x: int(x['image_id']))
    
    new = []
    for item in final:
        #print(item['final_score'])
        image_id = item["image_id"]
        caption = item["caption"]
        new_annotation = {"image_id": image_id, "caption": caption}
        new.append(new_annotation)
    new_data = {'annotations': new}
    folder_path = f'/path/to/test_dataset/selector/nhead{args.nhead}_res{args.residual}_LayerNum{args.layer_num}_featsize{args.feat_size}_num{args.data_num}/cc_sbu_align/'

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    image_path = '/path/to/cc_sbu_align/image'
    image_folder = 'image'  
    target_folder = folder_path + image_folder 
    shutil.rmtree(target_folder, ignore_errors=True)
    os.makedirs(target_folder, exist_ok=True)
    image_ids = [annotation['image_id'] for annotation in new]
    for image_id in image_ids:
        image_filename = f'{image_id}.jpg'
        source_path = os.path.join(image_path, image_filename)
        target_path = os.path.join(target_folder, image_filename)
        shutil.copyfile(source_path, target_path)
    
    with open(os.path.join(folder_path, 'filter_cap.json'), 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)
    print(folder_path)

if __name__ == "__main__":
    args = parse_args()
    main(args)