import random
import json

from PIL import Image

import os
import shutil


with open('clustering_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

image_folder = '/path/to/cc_sbu_align/image' 
for folder_name, image_names in data.items():
    # Create a new directory with the folder_name
    folder_name = '/img/' + folder_name
    os.makedirs(folder_name, exist_ok=True)
     
    # Move the corresponding image files to the new directory
    for image_name in image_names:
        image_filename = f'{image_name}.jpg'
        src_path = os.path.join(image_folder, image_filename)
        dst_path = os.path.join(folder_name, image_filename)
        shutil.copyfile(src_path, dst_path)

