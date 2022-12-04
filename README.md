# KBCIN

* Code for [AAAI 2023](https://aaai.org/Conferences/AAAI-23/) accepted paper titled "Knowledge-Bridged Causal Interaction Network for Causal Emotion Entailment"
* Weixiang Zhao, Yanyan Zhao, Zhuojun Li, Bing Qin.

## Requirements
* Python 3.7
* PyTorch 1.8.2
* Transformers 4.12.3
* CUDA 11.1

## Preparation

### Preprocessed Features
You can download the extracted utterance features and commonsense knowledge we used from:
https://pan.baidu.com/s/1n0RbyztvupFLFWe80vr7kg  提取码:qeqt

and place them into ./data

## Training
You can train the models with the following codes:

`python train.py --hidden_dim 300 --add_emotion --use_pos --use_emo_csk --use_act_csk --use_event_csk --lr 4e-5`
