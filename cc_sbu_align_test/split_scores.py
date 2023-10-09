import json
import argparse
import os
import shutil

def main():

    with open('full_score_data.json', 'r', encoding='utf-8') as f:
        data_origin = json.load(f)
    with open('gpt4score/gpt_final_score.json', 'r', encoding='utf-8') as f:
        gpt_data = json.load(f)
    origin_annotations = data_origin['annotations']
    gpt_annotations = gpt_data['annotations']
    new_annotations = []

    gpt_dic = {annotation['image_id']: annotation['gpt_score'] for annotation in gpt_annotations}
    clip_dic = {annotation['image_id']: annotation['clip_score'] for annotation in origin_annotations}
    reward_dic = {annotation['image_id']: annotation['reward_score'] for annotation in origin_annotations}
    len_dic = {annotation['image_id']: annotation['length_score'] for annotation in origin_annotations}

    clip = []
    gpt = []
    reward = []
    length = []

    for num in range(0,30):
        dir = f'/path/to/test_dataset/split/{num}/cc_sbu_align/'
        
        file = dir + 'filter_cap.json'
        with open(file, 'r', encoding='utf-8') as f:
            part = json.load(f)
        data = part['annotations']
        clipscore, rewardscore, lengthscore, gptscore = [],[],[],[]
        for i in range(len(data)):
            image_id = data[i]["image_id"]
            clipscore.append(clip_dic.get(image_id))
            rewardscore.append(reward_dic.get(image_id))
            lengthscore.append(len_dic.get(image_id))
            gptscore.append(gpt_dic.get(image_id))
        
        clip.append(clipscore)
        reward.append(rewardscore)
        length.append(lengthscore)
        gpt.append(gptscore)
    final = {'clip': clip, 'reward': reward, 'length': length, 'gpt': gpt}
    with open('fitting.json', 'w', encoding='utf-8') as f:
        json.dump(final, f, indent=4, ensure_ascii=False)
   

if __name__ == "__main__":
    main()