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

def get_csk_feature(model, tokenizer, context, relations):
    map1 = [{}, {}, {}, {}, {}]
    for id, conv in tqdm(context.items()):
        list1 = [[], [], [], [], []] 
        for utterance in conv:
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
        for k, v1 in enumerate(map1):
            v1[id] = list1[k]
    return map1

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
    relations_social = ["xReact", "xWant", "xIntent", "oReact", "oWant"]
    relations_event = ["isAfter", "HasSubEvent", "isBefore", "Causes", "xReason"]
    
    for split in ["train", "test", "valid"]:
        print ("\tSplit: {}".format(split))
        target_context, speaker, cause_label, \
        emotion_label, ids = pickle.load(open('./dailydialog_' + split + '.pkl', 'rb'), encoding='latin1')
        map1 = get_csk_feature(model, tokenizer, target_context, relations_social)
        pickle.dump(map1, open('./dailydialog_csk_social_' + split + '.pkl', "wb"))

    for split in ["train", "test", "valid"]:
        print ("\tSplit: {}".format(split))
        target_context, speaker, cause_label, \
        emotion_label, ids = pickle.load(open('./dailydialog_' + split + '.pkl', 'rb'), encoding='latin1')
        map1 = get_csk_feature(model, tokenizer, target_context, relations_event)
        pickle.dump(map1, open('./dailydialog_csk_event_' + split + '.pkl', "wb"))