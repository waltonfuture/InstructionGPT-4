import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from model.models import Selector
from torch.utils.data import TensorDataset, DataLoader
import json
import numpy as np
import random
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Selector")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--feat_size", type=int, default=6)
    parser.add_argument("--layer_num", type=int, default=2)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--concat", type=int, default=0) 
    parser.add_argument("--nhead", type=int, default=1)
    parser.add_argument("--residual", type=int, default=0) 
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
            feat_size = features_size,
            layer_num = args.layer_num,
            dimens = args.dimension,
            concat = args.concat,
            nhead = args.nhead,
            residual = args.residual
            )

    with open("/path/to/cc_sbu_align_test/fitting.json", "r") as file:
        data = json.load(file)

    clip_scores = torch.tensor(data["clip"])
    reward_scores = 10 * torch.tensor(data["reward"])
    length_scores = 10 * torch.tensor(data["length"])
    gpt_scores = torch.tensor(data["gpt"])

    labels = [30.86, 31.02, 29.51, 30.57, 30.95, 30.69, 30.45, 30.55, 31.51, 31.20, 31.36, 31.03, 31.69, 31.69, 31.56, 30.09, 29.93, 30.38, 31.38, 30.96, 31.36, 30.67, 31.33, 31.37, 29.92, 30.15, 29.86, 30.64, 30.31, 31.64] # replace it with the real label
    labels = torch.tensor(labels)  

    image_matrixes = torch.load(f'dataset/image_matrix.pt')
    text_matrixes = torch.load(f'dataset/text_matrix.pt')
  
    X = torch.stack((clip_scores, reward_scores, length_scores, gpt_scores), dim=2)
    

    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    dataset = TensorDataset(image_matrixes, text_matrixes, X, labels)
    train_loader = DataLoader(dataset, batch_size=30, shuffle=True)
 

    model.train()
    best_loss = float('inf')  
    best_model_weights = None

    for epoch in range(args.epoch):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0): 
            image_feats, text_feats, input_scores, labels = data 
            labels = labels.view(-1, 1)
            optimizer.zero_grad()
            outputs = model(instruction_logits=input_scores, image_feats=image_feats, text_feats=text_feats)

            loss = criterion(torch.mean(outputs['scores'], dim=1), labels.float()) 
            loss.backward()

            optimizer.step()
    
            running_loss += loss.item()
            
        print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 10}')
        running_loss = 0.0
        if running_loss < best_loss:
            best_loss = running_loss
            best_model_weights = model.state_dict()

    print('Finished Training')
  
    save_path = f'ckpt/nhead{args.nhead}_res{args.residual}_LayerNum{args.layer_num}_featsize{args.feat_size}_epoch{args.epoch}'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(best_model_weights, os.path.join(save_path, 'best_model.pth'))

if __name__ == "__main__":
    args = parse_args()
    main(args)