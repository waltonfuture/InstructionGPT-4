import json

with open('gpt4score/gpt4_score_new.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

annotations = data['annotations']

new_annotations = []

for annotation in annotations:
    gpt_score_str = annotation['gpt_score'][:2]
    try:
        gpt_score = int(gpt_score_str)
    except ValueError:
        # Handle the case where conversion fails and set gpt_score to 60
        gpt_score = 60
    if gpt_score == 10:
        gpt_score = 100
    new_annotation = {
        "image_id": annotation["image_id"],
        "caption": annotation["caption"],
        "gpt_score": gpt_score,
    }
    new_annotations.append(new_annotation)

new_data = {'annotations': new_annotations}

with open('gpt4score/gpt_final_score.json', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, indent=4, ensure_ascii=False)