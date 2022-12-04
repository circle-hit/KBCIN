import numpy
import pandas as pd
from tqdm import tqdm, trange
import json
import pickle
from transformers import RobertaTokenizer, BertTokenizer
import torch
from torch.nn.utils.rnn import pad_sequence

def get_new_input_data(split=None): 
    ori_data = json.load(open('../RECCON/data/original_annotation/dailydialog_' + split + '.json', 'r', encoding='utf-8'))
    emotion_dict = {'happiness': 0, 'neutral': 1, 'anger': 2, 'sadness': 3, 'fear': 4, 'surprise': 5, 'disgust': 6}
    
    emotion = {}
    target_context = {}
    speaker = {}
    cause_label = {}
    ids = []
    
    for id, v in tqdm(ori_data.items()):
        conversation = v[0]
        cnt = 1
        for item in conversation:
            if item['emotion'] == 'neutral':
                continue
            turn = item['turn']
            target = item['utterance']
            causal_utterances = item.get('expanded emotion cause evidence', None)
            if causal_utterances == None:
                continue
            cause_labels = [0] * turn
            for index in causal_utterances:
                if index != 'b' and index <= turn:
                    cause_labels[index-1] = 1
                
            context = []
            speaker_list = []
            emo = []
            for i in range(turn):
                cur_emo = conversation[i]['emotion']
                if cur_emo == 'sad':
                    cur_emo = 'sadness'
                if cur_emo == 'surprised':
                    cur_emo = 'surprise'
                if cur_emo == 'happy' or cur_emo == 'excited':
                    cur_emo = 'happiness'
                if cur_emo == 'angry':
                    cur_emo = 'anger'
                emo.append(emotion_dict[cur_emo])
                speaker_list.append(conversation[i]['speaker'])
                context.append(conversation[i]['utterance'])
            
            ids.append(id + '_' + str(cnt))
            target_context[id + '_' + str(cnt)] = context
            speaker[id + '_' + str(cnt)] = speaker_list
            cause_label[id + '_' + str(cnt)] = cause_labels
            emotion[id + '_' + str(cnt)] = emo
            cnt += 1
    
    pickle.dump([target_context, speaker, cause_label, emotion, ids], open('./data/dailydialog_' + split + '.pkl', 'wb'))

def data_statistics(split=None):
    target_context, speaker, cause_label, emotion, ids = pickle.load(open('./data/dailydialog_' + split + '.pkl', 'rb'), encoding='latin1')
    pos_cnt, neg_cnt = 0, 0
    total = cause_label.values()
    for item in total:
        for i in item:
            if i == 0:
                neg_cnt += 1
            else:
                pos_cnt += 1
    print("For " + split + ":")
    print("Pos Pairs:", pos_cnt)
    print("Neg Pairs:", neg_cnt)
    print("Max Length of Conversation:", max([len(item) for item in speaker.values()]))

def get_speaker_mask(totalIds, speakers):
    intra_masks, inter_masks = {}, {}
    for i in trange(len(totalIds)):
        id = totalIds[i]
        cur_speaker_list = speakers[id]
        cur_intra_mask = numpy.zeros((len(cur_speaker_list), len(cur_speaker_list)))
        cur_inter_mask = numpy.zeros((len(cur_speaker_list), len(cur_speaker_list)))
        target_speaker = cur_speaker_list[-1]
        target_index = len(cur_speaker_list) - 1
        
        cur_intra_mask[target_index][target_index] = 1
        for j in range(len(cur_speaker_list) - 1):
            if cur_speaker_list[j] == target_speaker:
                cur_intra_mask[target_index][j] = 1
            else:
                cur_inter_mask[target_index][j] = 1

            # 全连接
            if j == 0:
                cur_intra_mask[j][j] = 1
            else:
                for k in range(j):
                    if cur_speaker_list[j] == target_speaker:
                        cur_intra_mask[j][k] = 1
                    else:
                        cur_inter_mask[j][k] = 1

        intra_masks[id] = cur_intra_mask
        inter_masks[id] = cur_inter_mask
    
    return intra_masks, inter_masks

def get_relative_position(totalIds, speakers):
    relative_position = {}
    thr = 31
    for i in trange(len(totalIds)):
        id = totalIds[i]
        cur_speaker_list = speakers[id]
        cur_relative_position = []
        target_index = len(cur_speaker_list) - 1
        for j in range(len(cur_speaker_list)):
            if target_index - j < thr:
                cur_relative_position.append(target_index - j)
            else:
                cur_relative_position.append(31)
        relative_position[id] = cur_relative_position
    return relative_position

def process_for_ptm(dataset, split, model_size='base'):
    target_context, speaker, cause_label, emotion, ids = pickle.load(open('./data/dailydialog_' + split + '.pkl', 'rb'), encoding='latin1')

    token_ids, attention_mask = {}, {}
    tokenizer = RobertaTokenizer.from_pretrained('roberta-' + model_size) #  "bert-base-uncased"
    
    print("Tokenizing Input Dialogs ...")
    for id, v in tqdm(target_context.items()):
        cur_token_ids, cur_attention_mask = [], []
        for utterance in v:
            encoded_output = tokenizer(utterance)
            tid = encoded_output.input_ids
            atm = encoded_output.attention_mask
            cur_token_ids.append(torch.tensor(tid, dtype=torch.long))
            cur_attention_mask.append(torch.tensor(atm))
        tk_id = pad_sequence(cur_token_ids, batch_first=True, padding_value=1)
        at_mk = pad_sequence(cur_attention_mask, batch_first=True, padding_value=0)
        token_ids[id] = tk_id
        attention_mask[id] = at_mk

    print("Generating Speaker Connections ...")
    intra_mask, inter_mask = get_speaker_mask(ids, speaker)
    
    print("Generating Relative Position ...")
    relative_position = get_relative_position(ids, speaker)
    
    pickle.dump([target_context, token_ids, attention_mask, speaker, cause_label, emotion, relative_position, intra_mask, inter_mask, ids], open('./data/dailydialog_features_roberta_ptm_' + split + '.pkl', 'wb'))

if __name__ == '__main__':
    dataset = 'dailydialog'
    for split in ['train', 'valid', 'test']:
        # get_new_input_data(split)
        process_for_ptm(dataset, split)
        data_statistics(split)