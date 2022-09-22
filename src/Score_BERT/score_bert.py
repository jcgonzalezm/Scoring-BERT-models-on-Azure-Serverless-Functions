#!/usr/bin/env python#!/usr/bin/env python
# coding: utf-8

import os
import sys
import random
import json

import numpy as np
import torch

from tqdm import tqdm
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from torch.utils.data import (DataLoader, SequentialSampler,TensorDataset)


class ScoreBert():
    """
    This object will hold our loaded model and tokenizer nad host our scoring procedure.
    That way we load everything into memory at start,
    """
    def __init__(self, loaded_model=None, tokenizer=None):
        # Prepare
        device_type = "cpu"
        self.device = torch.device(device_type)

        # Seeds
        random.seed(100)
        np.random.seed(100)
        torch.manual_seed(100)
        if device_type == "cuda":
            torch.cuda.manual_seed_all(100)

        # Task Setup
        self.eval_batch_size = 64 #example
        self.label_list = self.processor.get_labels()
        self.num_labels = len(self.label_list)
        self.model = loaded_model
        self.tokenizer = tokenizer

        self.model.to(self.device)

    def score(self, text):
        """
        Scoring using the BERT Classification Model
        You will need to paste in here you entire score process
        This is a pseudo code only for ilustrative purposes
        """

        eval_features = self.tokenizer
        eval_features = convert_to_features(text) #Where all preprocessing its done 
        

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.eval_batch_size)

        self.model.eval()
        preds = []

        for input_ids, input_mask, segment_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask, labels=None)

            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

        preds = preds[0]

        return preds
        
def init():

    global mybert
    model_base_path = os.path.abspath(os.getcwd())

    def create_obj_scorebert(model_name, num_labels, tokenizer):
 
        model_dir = os.path.join(model_base_path,'Score_BERT/Models',model_name)
        model_state_dict = torch.load(model_dir, map_location="cpu")
        cache_dir = ""
        model = BertForSequenceClassification.from_pretrained(bert_model,
                        state_dict=model_state_dict, 
                        cache_dir=cache_dir,
                        num_labels=num_labels)
        mybert = ScoreBert(model, tokenizer=tokenizer)
        return mybert
    
    bert_model="bert-base-multilingual-cased"
    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case='False')

    mybert = create_obj_scorebert('bert_categories.bin', num_labels=24, tokenizer=tokenizer)

def run(json_input):
    
    data = json.loads(json_input)["data"]
    response = mybert.score(data)       
    return response

if __name__!="__main__":
    init()