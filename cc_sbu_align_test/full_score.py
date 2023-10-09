import random
import json
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from PIL import Image
import open_clip
import os
import shutil
import torch.nn as nn
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" 
device0 = torch.device("cuda:0")
device1 = torch.device("cuda:1")


# calculate the matching score
clip_tokenizer = open_clip.get_tokenizer('ViT-bigG-14')
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k')
clip_model = clip_model.cuda()

def get_clip_score(image, text):
    image = preprocess(image).unsqueeze(0).cuda()
    image_features = clip_model.encode_image(image)
    image_features /= image_features.norm(p=2, dim=-1, keepdim=True)

    text_features = []
    sentences = clip_tokenizer(text.split('.')[:10])
    text_features = clip_model.encode_text(sentences.cuda())
    text_features /= text_features.norm(p=2, dim=-1, keepdim=True)
    text_features = text_features.mean(dim=0, keepdim=True)

    similarity = (100.0 * image_features @ text_features.T).sum(dim=-1)
    return round(similarity.item(),4)


# human preference score
reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
rank_model, tokenizer = AutoModelForSequenceClassification.from_pretrained(reward_name), AutoTokenizer.from_pretrained(reward_name)
rank_model = rank_model.to(device1)

def get_reward_score(question, answer):
    inputs = tokenizer(question, answer, return_tensors='pt').to(device1)
    score = rank_model(**inputs).logits[0].cpu().detach()
    return round(score.item(),4)

max_caption_length = 1009 # minigpt4

def caption_length_score(answer):
    return 10 * len(answer) / 1009


with open('../cc_sbu_align/filter_cap.json', 'r', encoding='utf-8') as f:
    data_origin = json.load(f)


with open('gpt4score/gpt_final_score.json', 'r', encoding='gb18030') as f:
    gpt_data = json.load(f)


origin_annotations = data_origin['annotations']
gpt_annotations = gpt_data['annotations']

new_annotations = []
with torch.no_grad():
    for i in range(len(origin_annotations)):
        image_id = origin_annotations[i]["image_id"]
        caption = origin_annotations[i]["caption"]
        
        question = "Describe this image in detail."
        new_annotation = {"image_id": image_id, "caption": caption}
     
        img = Image.open(f"../cc_sbu_align/image/{image_id}.jpg") 
        new_annotation["clip_score"] = get_clip_score(img, caption)
        new_annotation["reward_score"] = get_reward_score(question, caption)
        new_annotation["length_score"] = caption_length_score(caption)
  
        new_annotations.append(new_annotation)



final = sorted(new_annotations, key=lambda x: int(x['image_id']))
new_data = {'annotations': final}

with open('full_score_data.json', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, indent=4, ensure_ascii=False)