import json
import torch
import pickle
import argparse
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from utils import use_task_specific_params, trim_batch
from tqdm import tqdm
import sys
sys.path.append("/users6/wxzhao/Empathy/CEM/src")


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class Comet:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        task = "summarization"
        use_task_specific_params(self.model, task)
        self.batch_size = 1
        self.decoder_start_token_id = None

    def generate(
            self, 
            queries,
            decode_method="beam", 
            num_generate=5, 
            ):

        with torch.no_grad():
            examples = queries

            decs = []
            for batch in list(chunks(examples, self.batch_size)):

                batch = self.tokenizer(batch, return_tensors="pt", truncation=True, padding="max_length").to(self.device)
                input_ids, attention_mask = trim_batch(**batch, pad_token_id=self.tokenizer.pad_token_id)

                summaries = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_start_token_id=self.decoder_start_token_id,
                    num_beams=num_generate,
                    num_return_sequences=num_generate,
                    )

                dec = self.tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                decs.append(dec)

            return decs


all_relations = [
    "AtLocation",
    "CapableOf",
    "Causes",
    "CausesDesire",
    "CreatedBy",
    "DefinedAs",
    "DesireOf",
    "Desires",
    "HasA",
    "HasFirstSubevent",
    "HasLastSubevent",
    "HasPainCharacter",
    "HasPainIntensity",
    "HasPrerequisite",
    "HasProperty",
    "HasSubEvent",
    "HasSubevent",
    "HinderedBy",
    "InheritsFrom",
    "InstanceOf",
    "IsA",
    "LocatedNear",
    "LocationOfAction",
    "MadeOf",
    "MadeUpOf",
    "MotivatedByGoal",
    "NotCapableOf",
    "NotDesires",
    "NotHasA",
    "NotHasProperty",
    "NotIsA",
    "NotMadeOf",
    "ObjectUse",
    "PartOf",
    "ReceivesAction",
    "RelatedTo",
    "SymbolOf",
    "UsedFor",
    "isAfter",
    "isBefore",
    "isFilledBy",
    "oEffect",
    "oReact",
    "oWant",
    "xAttr",
    "xEffect",
    "xIntent",
    "xNeed",
    "xReact",
    "xReason",
    "xWant",
    ]


def get_knowledge(model, split):
    
    target_context, speaker, cause_label, \
    emotion_label, ids = pickle.load(open('/users6/wxzhao/ERC/Emo_Cau_Con/data/new_fold1/dailydialog_' + split + '.pkl', 'rb'), encoding='latin1')
    f_out = open('/users6/wxzhao/ERC/Emo_Cau_Con/data/new_fold1/dailydialog_csk_visual_' + split + '.txt', 'w', encoding='utf-8')
    knowledge_set = {}
    # relation_set = ["oEffect", "oReact", "oWant", "xEffect", 
    #                 "xIntent", "xNeed", "xReact", "xReason", "xWant",
    #                 "HasSubEvent", "isAfter", "Causes"]
    relation_set = ["xReact", "xWant", "xIntent", "oReact", "oWant"]
    # relation_set = ["oReact", "xReact"]
    count = 1
    for id, conv in tqdm(target_context.items()):
        conv_knowledge = []
        for utterance in conv:
            # print(f'{count} processed')
            utter_knowledge = {}
            # utter_knowledge = 'this person feels'
            queries = []
            # utterance = utter['utterance']
            for r in relation_set:
                query = "{} {} [GEN]".format(utterance, r)
                queries.append(query)
            results = model.generate(queries, decode_method="beam", num_generate=5)
            for relation, result in zip(relation_set, results):
                utter_knowledge[relation] = ' ==sep== '.join(result)
                if count < 500:
                    outline = utterance + ' --relation-- ' + relation + ' --output-- ' +  ' ==sep== '.join(result)
                    f_out.write(outline + '\n')
                # if relation == 'oReact':
                #     utter_knowledge = utter_knowledge + result[0] + ' and' + result[1] + ' . '
                # else:
                #     utter_knowledge = utter_knowledge + 'others feels' + result[0] + ' and ' + result[1] + ' . '
            conv_knowledge.append(utter_knowledge)
            if count < 500:
                f_out.write('\n')
            count += 1
        knowledge_set[id] = conv_knowledge
    return knowledge_set

def get_knowledge_ed(model, data):
    for i in range(3):
        print("[situation]:", " ".join(data["situation"][i]))
        print("[emotion]:", data["emotion"][i])
        print("[context]:", [" ".join(u) for u in data["context"][i]])
        print("[target]:", " ".join(data["target"][i]))
        print(" ")
    print(stop)
    f_out = open('/users6/wxzhao/ERC/Emo_Cau_Con/data/new_fold1/dailydialog_csk_visual_' + split + '.txt', 'w', encoding='utf-8')
    knowledge_set = {}
    # relation_set = ["oEffect", "oReact", "oWant", "xEffect", 
    #                 "xIntent", "xNeed", "xReact", "xReason", "xWant",
    #                 "HasSubEvent", "isAfter", "Causes"]
    relation_set = ["xReact", "xWant", "xIntent", "oReact", "oWant"]
    # relation_set = ["oReact", "xReact"]
    count = 1
    for id, conv in tqdm(target_context.items()):
        conv_knowledge = []
        for utterance in conv:
            # print(f'{count} processed')
            utter_knowledge = {}
            # utter_knowledge = 'this person feels'
            queries = []
            # utterance = utter['utterance']
            for r in relation_set:
                query = "{} {} [GEN]".format(utterance, r)
                queries.append(query)
            results = model.generate(queries, decode_method="beam", num_generate=5)
            for relation, result in zip(relation_set, results):
                utter_knowledge[relation] = ' ==sep== '.join(result)
                if count < 500:
                    outline = utterance + ' --relation-- ' + relation + ' --output-- ' +  ' ==sep== '.join(result)
                    f_out.write(outline + '\n')
                # if relation == 'oReact':
                #     utter_knowledge = utter_knowledge + result[0] + ' and' + result[1] + ' . '
                # else:
                #     utter_knowledge = utter_knowledge + 'others feels' + result[0] + ' and ' + result[1] + ' . '
            conv_knowledge.append(utter_knowledge)
            if count < 500:
                f_out.write('\n')
            count += 1
        knowledge_set[id] = conv_knowledge
    return knowledge_set

def decode_knowledge_from_comet():
    # sample usage decode
    print("model loading ...")
    comet = Comet("./comet-atomic_2020_BART")
    comet.model.zero_grad()
    print("model loaded")

    data_tra, data_val, data_tst, vocab = pickle.load(open('/users6/wxzhao/Empathy/CEM/data/ED/dataset_preproc.p', "rb"))
    for i in range(3):
        print("[situation]:", " ".join(data_tra["situation"][i]))
        print("[emotion]:", data_tra["emotion"][i])
        print("[context]:", [" ".join(u) for u in data_tra["context"][i]])
        print("[target]:", " ".join(data_tra["target"][i]))
        print(" ")
    # print('train data')
    # train_knowledge = get_knowledge(comet, 'train')
    # print('dev data')
    # dev_knowledge = get_knowledge(comet, 'valid')
    print('test data')
    test_knowledge = get_knowledge_ed(comet, data_tst)

    # pickle.dump(train_knowledge, open('/users6/wxzhao/ERC/Emo_Cau_Con/data/new_fold1/dailydialog_csk_train.pkl', 'wb'))
    # pickle.dump(dev_knowledge, open('/users6/wxzhao/ERC/Emo_Cau_Con/data/new_fold1/dailydialog_csk_valid.pkl', 'wb'))
    # pickle.dump(test_knowledge, open('/users6/wxzhao/ERC/Emo_Cau_Con/data/new_fold1/dailydialog_csk_test.pkl', 'wb'))

def get_csk_feature(data):
    csk_features = []
    for conversation in tqdm(data):
        list1 = [[], [], [], [], [], [], [], []]
        cur_csk = {}
        for utterance in conversation:
            queries = []
            for r in relations:
                queries.append("{} {} [GEN]".format(utterance, r))
            with torch.no_grad():
                batch = tokenizer(queries, return_tensors="pt", truncation=True, padding="max_length").to(device)
                input_ids, attention_mask = trim_batch(**batch, pad_token_id=tokenizer.pad_token_id)
                out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                activations = out['decoder_hidden_states'][-1][:, 0, :].detach().cpu().numpy()
                for k, l1 in enumerate(list1):
                    l1.append(activations[k])
        for i, rel in enumerate(relations):
            cur_csk[rel] = list1[i]
        csk_features.append(cur_csk)
    return csk_features

if __name__ == "__main__":

    model_path = "./comet-atomic_2020_BART/"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).cuda()
    device = str(model.device)
    print(device)
    use_task_specific_params(model, "summarization")
    model.zero_grad()
    model.eval()
    
    batch_size = 8
    # relations = ["xReact", "xWant", "xIntent", "oReact", "oWant", "isAfter", "HasSubEvent", "isBefore", "Causes", "xReason"]
    # relations_event = ["isAfter", "HasSubEvent", "isBefore", "Causes", "xReason"]
    relations = ["xIntent", "xNeed", "xWant", "xEffect", "xReact", "oWant", "oEffect", "oReact"]
    
    
    with open("your_data.txt", "r", encoding="utf-8") as f:
        df_trn = f.read().split("\n")
    csk_feature = get_csk_feature(df_trn)
    