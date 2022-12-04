import numpy
import torch
from torch.nn.modules import padding
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd

def pad_matrix(matrix, padding_index=0):
    max_len = max(i.size(0) for i in matrix)
    batch_matrix = []
    for item in matrix:
        item = item.numpy()
        batch_matrix.append(numpy.pad(item, ((0, max_len-len(item)), (0, max_len-len(item))), 'constant', constant_values=(padding_index, padding_index)))
    return batch_matrix

class DailyDialogRobertaCometDataset(Dataset):

    def __init__(self, split):
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
        self.target_context, self.token_ids, self.attention_mask, self.speakers, self.cause_labels, self.emotion_label, \
        self.relative_position, self.intra_mask, self.inter_mask, self.Ids = pickle.load(open('./data/new_fold1/dailydialog_features_roberta_ptm_' + split + '.pkl', 'rb'), encoding='latin1')
        
        self.isAfter, self.HasSubEvent, self.isBefore, self.Causes, self.xReason \
        = pickle.load(open('./data/new_fold1/dailydialog_csk_event_' + split + '.pkl', 'rb'), encoding='latin1')

        self.xReact, self.xWant, self.xIntent, self.oReact, self.oWant \
        = pickle.load(open('./data/new_fold1/dailydialog_csk_social_' + split + '.pkl', 'rb'), encoding='latin1')

        self.keys = [x for x in self.Ids]
        self.len = len(self.keys)
        
    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.isBefore[vid]),\
               torch.FloatTensor(self.isAfter[vid]),\
               torch.FloatTensor(self.xWant[vid]),\
               torch.FloatTensor(self.xReact[vid]),\
               torch.FloatTensor(self.oWant[vid]),\
               torch.FloatTensor(self.oReact[vid]),\
               torch.FloatTensor([[1,0] if x=='A' else [0,1] for x in self.speakers[vid]]),\
               torch.FloatTensor([1]*len(self.cause_labels[vid])),\
               torch.LongTensor(self.cause_labels[vid]),\
               torch.LongTensor(self.emotion_label[vid]),\
               torch.LongTensor(self.relative_position[vid]),\
               torch.FloatTensor(self.intra_mask[vid]),\
               torch.FloatTensor(self.inter_mask[vid]),\
               self.attention_mask[vid],\
               self.token_ids[vid],\
               self.target_context[vid], \
               self.speakers[vid], \
               self.Ids[index]
               
    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        intra_mask = torch.FloatTensor(pad_matrix(dat[11]))
        inter_mask = torch.FloatTensor(pad_matrix(dat[12]))
        return [pad_sequence(dat[i]) if i<6 else pad_sequence(dat[i], True) if i<10 else pad_sequence(dat[i], True, padding_value=31) if i<11 else intra_mask if i<12 else inter_mask if i<13  else dat[i] for i in dat]