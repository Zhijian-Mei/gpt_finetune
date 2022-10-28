import argparse
import os
import random
from pprint import pprint

import numpy as np
import pandas as pd
import torch.cuda
from tqdm import trange
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
from transformers import BartForConditionalGeneration,BartTokenizer
# from data_util import *
from transformers import Trainer, TrainingArguments
from torch.optim import AdamW
# from transformers.optimization import AdamW
from torch.utils.data import Dataset, random_split,DataLoader
from sklearn.model_selection import train_test_split
from metrices import *
# from openprompt import PromptForGeneration
# from openprompt.prompts import ManualTemplate
import json

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-train_path', type=str, default=None)
    parser.add_argument('-train_batch', type=int, default=2)
    parser.add_argument('-eval_batch',type=int,default=1)
    parser.add_argument('-gpu', type=str, default='0')

    args = parser.parse_args()
    return args

def get_prompt_complicated(attribute_dic,mask_token):
    prompt_dic = {
        'Type':f'The type of the item is {mask_token}',
        'Silhouette':f'The silhouette of the item is '






    }
    return prompt_dic

def get_prompt_simple(df,mask_token):
    inputs = []
    outputs = []
    attributes = df['attributes']
    titles = df['title']
    descriptions = df['description']
    for i in range(len(df)):
        input_base = ''
        if not pd.isna(titles[i]):
            input_base += titles[i] + '; '
        if not pd.isna(descriptions[i]):
            input_base += descriptions[i][2:-2] + '; '
        if not pd.isna(attributes[i]):
            item_attributes = json.loads(df['attributes'][i])
            for key in item_attributes.keys():
                masked_prompt = f'the {key} of the item is {mask_token}'
                prompt = f'the {key} of the item is {random.choice(item_attributes[key])}'
                temp_input = input_base + masked_prompt
                temp_output = input_base + prompt
                inputs.append(temp_input)
                outputs.append(temp_output)
            continue
        inputs.append(input_base)
        outputs.append(input_base)

    return inputs,outputs

class MyDataset(Dataset):
    def __init__(self,df,tokenizer,batch_size=32,train = False):
        self._x = []
        self._y = []
        self.train = train
        inputs = df['inputs'].values
        outputs = df['outputs'].values
        for i in trange(0,len(df),batch_size):
            batch_x = inputs[i:i+batch_size]
            batch_y = outputs[i:i+batch_size]
            encoded_batch_x = tokenizer.batch_encode_plus(batch_x, truncation=True,
                                                          max_length=1024, padding="longest",
                                                          return_tensors='pt')

            self._x.append(encoded_batch_x)
            if not train:
                self._y.append(batch_y)
            else:
                encoded_batch_y = tokenizer.batch_encode_plus(batch_y, truncation=True,
                                                          max_length=1024, padding="longest",
                                                          return_tensors='pt')
                self._y.append(encoded_batch_y)
        self._len = len(self._x)

    def __getitem__(self, index):
        return self._x[index],self._y[index]

    def __len__(self):
        return self._len


# # make sure GPT2 appends EOS in begin and end
# def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
#     outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
#     return outputs


if __name__ == '__main__':
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_batch_size = 2
    eval_batch_size = 4

    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(device)
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    df = pd.read_csv('../MAVE_filtered.csv')[['title','description','attributes']]
    inputs,outputs = get_prompt_simple(df,tokenizer.mask_token)
    df = pd.DataFrame()
    df['inputs'] = inputs
    df['outputs'] = outputs
    train, eval = train_test_split(df, train_size=0.7, test_size=0.3,random_state=48)
    eval, test = train_test_split(eval, train_size=0.5, test_size=0.5,random_state=48)

    trainDataset = MyDataset(train,tokenizer,batch_size=train_batch_size,train=True)
    evalDataset = MyDataset(eval,tokenizer,batch_size=eval_batch_size)
    testDataset = MyDataset(test,tokenizer,batch_size=eval_batch_size)


    # GPT2Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # # set pad_token_id to unk_token_id -> be careful here as unk_token_id == eos_token_id == bos_token_id
    # tokenizer.pad_token = tokenizer.eos_token
    #
    #
    # model = GPT2LMHeadModel.from_pretrained('gpt2', return_dict=True).to(device)

    optimizer = AdamW(
        [
            {'params': model.parameters(), 'lr': 1e-5},
        ])
    initial_bleu = -1
    # train
    epoch = 3
    for ep in range(epoch):
        model.train()
        total_loss = 0
        for i in trange(len(trainDataset)):
            batch = trainDataset[i]
            x = batch[0].to(device)
            y = batch[1].to(device)
            optimizer.zero_grad()
            outputs = model(**x, labels=y['input_ids'])
            loss = outputs.loss
            # logits = outputs.logits
            # indices = torch.argmax(logits, dim=2)
            # print(tokenizer.batch_decode(indices, skip_special_tokens=True))
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f'Average loss epoch {ep + 1} = {total_loss/len(trainDataset)}')

        # evaluation
        torch.save(f'checkpoints/bart_epoch-{ep + 1}.pt')

        model.eval()
        references = []
        candidates = []
        for i in trange(len(evalDataset)):
            batch = evalDataset[i]
            x = batch[0].to(device)
            y = batch[1]
            # print(y)
            # quit()
            references += y
            # outputs = model(**eval_batch, labels=eval_batch['input_ids'])
            # logits = outputs.logits
            # indices = torch.argmax(logits, dim=2)
            # candidates += tokenizer.batch_decode(indices, skip_special_tokens=True)
            # print(tokenizer.batch_decode(indices, skip_special_tokens=True))
            sample_outputs = model.generate(
                input_ids=x.input_ids,
                attention_mask=x.attention_mask,
                max_length =128,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.pad_token_id,
                num_beams=2,
                early_stopping=True
            )


            candidates += tokenizer.batch_decode(sample_outputs,skip_special_tokens=True)
            # print(tokenizer.batch_decode(x['input_ids'],skip_special_tokens=True))
            # print(tokenizer.batch_decode(sample_outputs,skip_special_tokens=True))
            # quit()
        # print(references)
        # print(candidates)

        assert len(references) == len(candidates)
        current_bleu = bleu(references=references,hypothesis=candidates)
        print(current_bleu)



