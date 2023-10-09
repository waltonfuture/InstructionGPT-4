import json

# Record the scores (clip, reward, length, gpt) for every data
with open('/cc_sbu_align_test/full_score_data.json', 'r', encoding='gb18030') as f:
    data_origin = json.load(f)
with open('/cc_sbu_align_test/gpt4score/gpt_final_score.json', 'r', encoding='gb18030') as f:
    gpt_data = json.load(f)
origin_annotations = data_origin['annotations']
gpt_annotations = gpt_data['annotations']

gpt_dic = {annotation['image_id']: annotation['gpt_score'] for annotation in gpt_annotations}
clip_dic = {annotation['image_id']: annotation['clip_score'] for annotation in origin_annotations}
reward_dic = {annotation['image_id']: annotation['reward_score'] for annotation in origin_annotations}
len_dic = {annotation['image_id']: annotation['length_score'] for annotation in origin_annotations}

clip = []
gpt = []
reward = []
length = []
with open('/path/to/cc_sbu_align/filter_cap.json', 'r', encoding='utf-8') as f:
    part = json.load(f)
data = part['annotations']
clipscore, rewardscore, lengthscore, gptscore = 0,0,0,0
for i in range(len(data)):
    image_id = data[i]["image_id"]
    clipscore = clip_dic.get(image_id)
    rewardscore = reward_dic.get(image_id)
    lengthscore = len_dic.get(image_id)
    gptscore = gpt_dic.get(image_id)
    clip.append(clipscore)
    reward.append(rewardscore)
    length.append(lengthscore)
    gpt.append(gptscore)
final = {'clip': clip, 'reward': reward, 'length': length, 'gpt': gpt}
with open('scores.json', 'w', encoding='utf-8') as f:
    json.dump(final, f, indent=4, ensure_ascii=False)