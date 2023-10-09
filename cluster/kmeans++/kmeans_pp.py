import random
import json
import torch
from PIL import Image
import open_clip
import os
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

clip_tokenizer = open_clip.get_tokenizer('ViT-bigG-14')
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k')
clip_model = clip_model.cuda()

def get_image_embed(image):
    image = preprocess(image).unsqueeze(0).cuda()
    image_features = clip_model.encode_image(image)
    return image_features


with open('/path/to/cc_sbu_align_test/filter_cap.json', 'r', encoding='utf-8') as f:
    data_origin = json.load(f)

origin_annotations = data_origin['annotations']

image_vectors = [] 
image_ids = []
new_annotations = []
with torch.no_grad():
    for i in range(len(origin_annotations)):
        image_id = origin_annotations[i]["image_id"]
        img = Image.open(f"/path/to/cc_sbu_align/image/{image_id}.jpg")
        img_embed = get_image_embed(img)
        image_vectors.append(img_embed)
        image_ids.append(image_id)


image_matrix = torch.cat(image_vectors, dim=0)
image_matrix = image_matrix.cpu().numpy()

n_clusters = 30

# Perform K-means++ clustering
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=0)
cluster_labels = kmeans.fit_predict(image_matrix)

clustered_images = {}
for i, label in enumerate(cluster_labels):
    image_id = image_ids[i]
    if label not in clustered_images:
        clustered_images[label] = []
    clustered_images[label].append(image_id)

for cluster_label, images in clustered_images.items():
    print(f"Cluster {cluster_label}: {len(images)} images")

clustered_images = {int(k): v for k, v in clustered_images.items()}

with open('clustering_results.json', 'w', encoding='utf-8') as f:
    json.dump(clustered_images, f, indent=4, ensure_ascii=False)
