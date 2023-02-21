import numpy as np, argparse, time, pickle, random, math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from dataloader import DailyDialogRobertaCometDataset
from model import MaskedBCELoss
from models.emotion_cause_model import KBCIN
from sklearn.metrics import f1_score, accuracy_score, classification_report
from transformers import AdamW, get_constant_schedule, get_linear_schedule_with_warmup
from tqdm import tqdm
import json

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_DailyDialog_loaders(batch_size=8, num_workers=0, pin_memory=False):
    trainset = DailyDialogRobertaCometDataset('train')
    validset = DailyDialogRobertaCometDataset('valid')
    testset = DailyDialogRobertaCometDataset('test')

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader

def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False):
    losses, preds, labels, masks, losses_sense  = [], [], [], [], []

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    with tqdm(total=int(len(dataloader) / args.accumulate_step), desc=f"Epoch {epoch+1}") as pbar:
        for step, data in enumerate(dataloader):
            if train:
                optimizer.zero_grad()

            bf, af, xW, xR, oW, oR, \
            qmask, umask, label, emotion_label, relative_position, \
            intra_mask, inter_mask, attention_mask, token_ids, \
            utterances, speaker, Ids = [data[i].cuda() if i<13 else data[i] for i in range(len(data))] if cuda else data
            attention_mask = [t.cuda() for t in attention_mask]
            token_ids = [t.cuda() for t in token_ids]

            log_prob, _, _ = model(token_ids, attention_mask, emotion_label+1, relative_position, intra_mask, inter_mask, bf, af, xW, xR, oW, oR, qmask, umask)

            # BCE loss
            lp_ = log_prob # [batch, seq_len]
            labels_ = label # [batch, seq_len]
            loss = loss_function(labels_, lp_, umask)

            if args.accumulate_step > 1:
                loss = loss / args.accumulate_step
    
            lp_ = log_prob.view(-1)
            labels_ = labels_.view(-1)
            pred_ = torch.gt(lp_.data, 0.5).long() # batch*seq_len
    
            preds.append(pred_.data.cpu().numpy())
            labels.append(labels_.data.cpu().numpy())
            masks.append(umask.view(-1).cpu().numpy())
            losses.append(loss.item()*masks[-1].sum())
            
            if train:
                total_loss = loss
                if args.fp16:
                    with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    total_loss.backward()
                
                if (step + 1) % args.accumulate_step == 0:
                    pbar.update(1)
                    optimizer.step()
                step += 1
            else:
                pbar.update(1)

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), float('nan'), [], [], [], float('nan'),[]

    avg_loss = round(np.sum(losses)/np.sum(masks), 4)

    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='macro')*100, 2)
    if train == False:
        reports = classification_report(labels,
                                        preds,
                                        target_names=['neg', 'pos'],
                                        sample_weight=masks,
                                        digits=4)
        return avg_loss, [avg_fscore], reports
    else:
        return avg_loss, [avg_fscore]

def save_badcase(model, dataloader, cuda, args):
    preds, labels = [], []
    scores, vids = [], []
    dialogs = []
    speakers = []
    conv_lens = []

    model.eval()
    dialog_id = 1
    f_out = open('./badcase/badcase_dd.txt', 'w', encoding='utf-8')
    print("Logging Badcase ...")
    for data in tqdm(dataloader):

        r1, r2, r3, r4, x1, x2, x3, x4, x5, x6, o1, o2, o3, \
        qmask, umask, label, emotion_label, relative_position, \
        intra_mask, inter_mask, attention_mask, token_ids, \
        edge_index, edge_type, utterances, speaker, Ids = [data[i].cuda() if i<20 else data[i] for i in range(len(data))] if cuda else data
        attention_mask = [t.cuda() for t in attention_mask]
        token_ids = [t.cuda() for t in token_ids]

        utterances = [u for u in utterances]
        speaker = [s for s in speaker]

        # print(speakers)
        log_prob = model(token_ids, attention_mask, emotion_label, relative_position, edge_index, edge_type, intra_mask, inter_mask, r1, r2, r3, r4, x1, x2, x3, x4, x5, x6, o1, o2, o3, qmask, umask)
        conv_len = torch.sum(umask != 0, dim=-1).cpu().numpy().tolist()
        # umask = umask.cpu().numpy().tolist()
        label = label.cpu().numpy().tolist() # (B, N)
        pred = torch.gt(log_prob.data, 0.5).long().cpu().numpy().tolist() # (B, N)
        preds += pred
        labels += label
        dialogs += utterances
        speakers += speaker
        conv_len = [item for item in conv_len]
        conv_lens += conv_len

        # finished here

    if preds != []:
        new_preds = []
        new_labels = []
        for i,label in enumerate(labels):
            for j in range(conv_lens[i]):
                new_labels.append(label[j])
                new_preds.append(preds[i][j])
    else:
        return

    cases = []
    for i,d in enumerate(dialogs):
        case = []
        for j,u in enumerate(d):
            case.append({
                'text': u,
                'speaker': speakers[i][j],
                'label': labels[i][j],
                'pred': preds[i][j]
            })
            f_out.write(str(dialog_id) + '\t')
            f_out.write(u + '\t')
            f_out.write(speakers[i][j] + '\t')
            f_out.write(str(labels[i][j]))
            f_out.write('\t')
            f_out.write(str(preds[i][j]) + '\n')
        cases.append(case)
        dialog_id += 1

    with open('badcase/dailydialog.json', 'w', encoding='utf-8') as f:
        json.dump(cases,f)

    avg_accuracy = round(accuracy_score(new_labels, new_preds) * 100, 2)
    avg_fscore = round(f1_score(new_labels, new_preds, average='macro') * 100, 2)
    print('badcase saved')
    print('test_f1', avg_fscore)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=4e-5, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.0003, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--accumulate_step', type=int, required=False, default=1)
    parser.add_argument('--weight_decay', type=float, required=False, default=3e-4)
    parser.add_argument('--scheduler', type=str, required=False, default='constant')
    parser.add_argument('--warmup_rate', type=float, required=False, default=0.06)
    parser.add_argument('--rec-dropout', type=float, default=0.3, metavar='rec_dropout', help='rec_dropout rate')
    parser.add_argument('--mlp_dropout', type=float, default=0.07, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch_size', type=int, default=8, metavar='BS', help='batch size')
    parser.add_argument('--speaker_num', type=int, default=9, metavar='SN', help='number of speakers')
    parser.add_argument('--epochs', type=int, default=40, metavar='E', help='number of epochs')
    parser.add_argument('--num_attention_heads', type=int, default=6, help='Number of output mlp layers.')
    parser.add_argument('--hidden_dim', type=int, default=100, metavar='HD', help='hidden feature dim')
    parser.add_argument('--emotion_dim', type=int, default=300, metavar='HD', help='hidden feature dim')
    parser.add_argument('--roberta_dim', type=int, default=1024, metavar='HD', help='hidden feature dim')
    parser.add_argument('--csk_dim', type=int, default=1024, metavar='HD', help='hidden feature dim')
    parser.add_argument('--seed', type=int, default=42, metavar='seed', help='seed')
    parser.add_argument('--norm', action='store_true', default=False, help='normalization strategy')
    parser.add_argument('--save', action='store_true', default=False, help='whether to save best model')
    parser.add_argument('--add_emotion', action='store_true', default=False, help='whether to use emotion info')
    parser.add_argument('--use_emo_csk', action='store_true', default=False, help='whether to use emo commonsense knowledge')
    parser.add_argument('--use_act_csk', action='store_true', default=False, help='whether to use act commonsense knowledge')
    parser.add_argument('--use_event_csk', action='store_true', default=False, help='whether to use event knowledge')
    parser.add_argument('--use_pos', action='store_true', default=False, help='whether to use position embedding')
    parser.add_argument('--rnn_type', default='GRU', help='RNN Type')
    parser.add_argument('--model_size', default='base', help='roberta-base or large')
    parser.add_argument('--model_type', type=str, required=False, default='v2')
    parser.add_argument('--conv_encoder', type=str, required=False, default='none')
    parser.add_argument('--rnn_dropout', type=float, required=False, default=0.5)
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
    parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1', help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].")

    args = parser.parse_args()
    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    cuda       = args.cuda
    n_epochs   = args.epochs
    batch_size = args.batch_size

    if args.add_emotion:
        emotion_num = 8 # 7 categories plus 1 padding
    else:
        emotion_num = 0

    global seed
    seed = args.seed
    for seed in [0, 1, 2, 3, 4]: # to reproduce results reported in the paper
        seed_everything(seed)

        model = KBCIN(args, emotion_num)
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(n, p.size())
                if len(p.shape) > 1:
                    torch.nn.init.xavier_uniform_(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)
        print ('DailyDialog RECCON Model.')
        
        if cuda:
            model.cuda()

        loss_function = MaskedBCELoss()

        train_loader, valid_loader, test_loader = get_DailyDialog_loaders(batch_size=batch_size, num_workers=0)
        
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

        scheduler_type = args.scheduler
        if scheduler_type == 'linear':
            num_conversations = len(train_loader.dataset)
            if (num_conversations * n_epochs) % (batch_size * args.accumulate_step) == 0:
                num_training_steps = (num_conversations * n_epochs) / (batch_size * args.accumulate_step)
            else:
                num_training_steps = (num_conversations * n_epochs) // (batch_size * args.accumulate_step) + 1
            num_warmup_steps = int(num_training_steps * args.warmup_rate)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
        else:
            scheduler = get_constant_schedule(optimizer)
        
        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
 
        valid_losses, valid_fscores = [], []
        test_fscores, test_losses, iemocap_test_fscores, iemocap_test_losses = [], [], [], []
        valid_reports, test_reports, iemocap_test_reports = [], [], []
        best_loss, best_label, best_pred, best_mask = None, None, None, None
        best_model = None
    
        max_valid_f1 = 0
        continue_not_increase = 0
        for e in range(n_epochs):
            increase_flag = False
            start_time = time.time()
            train_loss, train_fscore = train_or_eval_model(model, loss_function, train_loader, e, optimizer, True)
            valid_loss, valid_fscore, valid_report = train_or_eval_model(model, loss_function, valid_loader, e)
            test_loss, test_fscore, test_report = train_or_eval_model(model, loss_function, test_loader, e)
            if valid_fscore[0] > max_valid_f1:
                max_valid_f1 = valid_fscore[0]
                best_model = model
                increase_flag = True
                if args.save:
                    torch.save(model.state_dict(), open('./save_dicts/best_model_{}'.format(str(seed)) + '.pkl', 'wb'))
                    print('Best Model Saved!')
            valid_losses.append(valid_loss)
            valid_fscores.append(valid_fscore)
            valid_reports.append(valid_report)
            test_losses.append(test_loss)
            test_fscores.append(test_fscore)
            test_reports.append(test_report)

            x = 'epoch: {}, train_loss: {}, fscore: {}, valid_loss: {}, fscore: {}, test_loss: {}, fscore: {}, time: {} sec'.format(e+1, train_loss, train_fscore, valid_loss, valid_fscore, test_loss, test_fscore, round(time.time()-start_time, 2))
            print (x)
            if increase_flag == False:
                continue_not_increase += 1
                if continue_not_increase >= 5:
                    print('early stop.')
                    break
            else:
                continue_not_increase = 0
        
        valid_fscores = np.array(valid_fscores).transpose()
        test_fscores = np.array(test_fscores).transpose()
        iemocap_test_fscores = np.array(iemocap_test_fscores).transpose()
        score1 = test_fscores[0][np.argmin(valid_losses)]
        score2 = test_fscores[0][np.argmax(valid_fscores[0])]
        score3 = test_fscores[0][np.argmax(test_fscores[0])]
        report_valid = test_reports[np.argmax(valid_fscores[0])]
        report_test = test_reports[np.argmax(test_fscores[0])]
        scores = [score1, score2]
        scores_val_loss = [score1]
        scores_val_f1 = [score2]
        scores_test_f1 = [score3]
        scores = [str(item) for item in scores]
        print ('Test Scores:')
        print('For RECCON-DD:')
        print('F1@Best Valid Loss: {}'.format(scores_val_loss))
        print('F1@Best Valid F1: {}'.format(scores_val_f1))
        print('F1@Best Test F1: {}'.format(scores_test_f1))
        print(report_valid)
        print(report_test)