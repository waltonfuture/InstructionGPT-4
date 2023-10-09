import random
import json
import shutil
import os
with open('/path/to/cc_sbu_align/filter_cap.json', 'r', encoding='utf-8') as f:
    data_origin = json.load(f)

origin_annotations = data_origin['annotations']

for i in range(30):
    save_dir = f'/path/to/test_dataset/split/{i}/cc_sbu_align/'
    image_folder = 'image'  
    target_folder = save_dir + image_folder 

    save_file = save_dir + 'filter_cap.json'
    new_annotations = []
    for i in range(len(origin_annotations)):
        image_id = origin_annotations[i]["image_id"]
        caption = origin_annotations[i]["caption"]
        image_file = image_id + '.jpg'
        target_path = os.path.join(target_folder, image_file)
        if os.path.exists(target_path):
            new_annotation = {"image_id": image_id, "caption": caption}
            new_annotations.append(new_annotation)

    new_data = {'annotations': new_annotations}
    with open(save_file, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)

