import random
import json

from PIL import Image

import os
import shutil

# 3439 / 30 = 114
with open('clustering_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

image_counts = {folder: len(image_names) for folder, image_names in data.items()}

total_image_count = len(data) * 114

excess_image_count = sum(max(0, count - 114) for count in image_counts.values())

if excess_image_count > 0:
    for folder in data.keys():
        if image_counts[folder] > 114:
            for target_folder in data.keys():
                if image_counts[target_folder] < 114:
                    move_count = min(114 - image_counts[target_folder], image_counts[folder] - 114)
                    if move_count > 0:
                        data[target_folder].extend(data[folder][:move_count])
                        data[folder] = data[folder][move_count:]
                        image_counts[folder] -= move_count
                        image_counts[target_folder] += move_count
                    if image_counts[folder] <= 114:
                        break

for folder, image_names in data.items():
    if len(image_names) > 114:
        data[folder] = image_names[:114]  
        print("done")

with open('modified.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4)