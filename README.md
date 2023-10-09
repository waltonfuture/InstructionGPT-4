# InstructionGPT-4: A 200-Instruction Paradigm for Fine-Tuning MiniGPT-4
[Lai Wei](https://waltonfuture.github.io/) , Zihao Jiang, [Weiran Huang](https://www.weiranhuang.com/), [Lichao Sun](https://lichao-sun.github.io/).

**Shanghai Jiao Tong University & Lehigh University**

<a href='https://mp.weixin.qq.com/s/s4Acec71v5oMlFkyhlCL_g'><img src='https://img.shields.io/badge/Project-Link-Green'></a>  <a href='https://arxiv.org/abs/2308.12067'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> <a href='https://huggingface.co/datasets/WaltonFuture/InstructionGPT-4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-blue'></a> 
<p align="center">
    <img src="images/instructiongpt4.png" width="30%"> <br>
</p>

## Introduction
- We introduce InstructionGPT-4, which is fine-tuned on a small dataset comprising only 200 examples, amounting to approximately 6% of the instruction-following data used in the alignment dataset for MiniGPT-4. 
- We first propose several metrics to access the quality of multimodal instruction data.
Based on these metrics, we present an effective and trainable data selector to automatically identify and filter low-quality vision-language data.
By employing this method, InstructionGPT-4 outperforms the original MiniGPT-4 on various evaluations.
- Our findings demonstrate that less but high-quality instruction tuning data is efficient to enable multimodal large language models to generate better output.

## Getting Started
### Prepare MiniGPT-4 Environment and Dataset
```bash
git clone https://github.com/Vision-CAIR/MiniGPT-4.git
```
Follow this [doc](https://github.com/Vision-CAIR/MiniGPT-4) to prepare the environment and download the cc_sbu_align dataset.

You also need to replace '/path/to' with your own path.
### Compute Indicators

#### 1. CLIP Score, Reward Score, Length Score, GPT Score
Fill in your openai key [here](cc_sbu_align_test/generate_gpt_score.py#L5).

Generate GPT Score:
```bash
cd cc_sbu_align_test
python generate_gpt_score.py
python convert_score.py
```
Generate CLIP Score, Reward Score and Length Score:
```bash
python full_score.py
```


#### 2. Multimodal Features
```bash
cd selector
python utils_image.py
python utils_text.py
python get_all_image_features.py
python get_all_text_features.py
```

### Compute Genuine Quality Labels
#### 1. Split into 30 subsets
```bash
cd cluster/kmeans++
python kmeans_pp.py
python average.py
python image_show.py
python split.py
python image2cap.py
```
#### 2. Evaluate to get the labels
Use each subset to fine-tune a new MiniGPT-4 model and follow this [doc](https://github.com/OpenGVLab/Multi-Modality-Arena) for evaluation. We choose GQA, IconQA, OKVQA and ScienceQA as the validation sets. 
### Data Selector
#### 1. Training
```bash
cd selector
bash train.sh
```

#### 2. Testing
Conduct clustering to ensure the diversity.
```bash
cd cluster/spectral
python spectral_clustering.py
python image_show.py
```
Select the final dataset for InstructionGPT-4
```bash
cd selector
bash eval_clus.sh
```
200 multimodal instruction data are selected [here](https://huggingface.co/datasets/WaltonFuture/InstructionGPT-4).

InstructionGPT-4 is fine-tuned from these 200 selected samples based on pre-trained MiniGPT-4.
## Acknowledgement

+ [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) The model architecture of InstructionGPT-4 follows MiniGPT-4. 
+ [LIMA](https://arxiv.org/abs/2305.11206) Our inspiration comes from LIMA: Less Is More for Alignment.
+ [Alpagasus](https://lichang-chen.github.io/AlpaGasus/) We follow Alpagasus to design the GPT Score.


If you're using InstructionGPT-4 in your research or applications, please cite using this BibTeX:
```bibtex
@article{wei2023instructiongpt,
  title={InstructionGPT-4: A 200-Instruction Paradigm for Fine-Tuning MiniGPT-4},
  author={Wei, Lai and Jiang, Zihao and Huang, Weiran and Sun, Lichao},
  journal={arXiv preprint arXiv:2308.12067},
  year={2023}
}
```