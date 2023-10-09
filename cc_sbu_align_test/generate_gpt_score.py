"""Call openai api."""
import json
import time
import openai
openai.api_key = ''
openai.organization = ''
openai.proxy = ''
def get_completion(prompt, model="gpt-4"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=1, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

prompt = f"""
We would like to request your feedback on the performance of an AI assistant. The assistant provides a caption based on an image and an instruction. Please rate according to the quality and variety of the caption to the instruction. Each assistant receives a score on a scale of 0 to 100, where a higher score indicates higher level of the quality and variety. Please first output a single line containing the value indicating the scores. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias. The instruction and caption are displayed following without image.

Instruction: [Instruction]
Caption: [Caption]
"""


with open('cc_sbu_align_test/filter_cap.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

output = []

instruction = "Describe this image in detail."
annotations = data['annotations']

max_retries = 20
for annotation in annotations:
    caption = annotation['caption']
    image_id = annotation['image_id']
 
    inputs = prompt.replace("[Instruction]", instruction).replace("[Caption]", caption)
    ok = True
    retries = 0
    while ok and retries <= max_retries:
        try:
            response = get_completion(inputs)
            ok = False
        except openai.error.OpenAIError as e:
            retries += 1
            print(f"Request failed. Retrying ({retries}/{max_retries})...")
            time.sleep(2 ** retries) # Exponential backoff delay
    if not ok:
        annotation['gpt_score'] = response
        output.append(annotation)
        new_data = {'annotations': output}
        with open('gpt4score/gpt4_score_new.json', 'w', encoding='utf-8') as f:
            json.dump(new_data, f, indent=4, ensure_ascii=False)
            f.write('\n')  
    else:
        print("stop!")
        break