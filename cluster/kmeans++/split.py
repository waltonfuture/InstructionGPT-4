import random
import json

from PIL import Image

import os
import shutil

with open('/path/to/cc_sbu_align_test/full_score_data.json', 'r', encoding='gb18030') as f:
    data_scores = json.load(f)

data_score = data_scores['annotations']

source_folder = "img"

target_base_folder = "/path/to/test_dataset/split/"

for folder_name in os.listdir(source_folder):
    target_folder = os.path.join(target_base_folder, folder_name, "cc_sbu_align","image")

    os.makedirs(target_folder, exist_ok=True)

    source_folder_path = os.path.join(source_folder, folder_name)
    for filename in os.listdir(source_folder_path):
        source_file = os.path.join(source_folder_path, filename)
        target_file = os.path.join(target_folder, filename)
        shutil.copyfile(source_file, target_file)

    print(f"copy {folder_name} to {target_folder} done")
