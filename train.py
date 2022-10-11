import argparse
import os

import pandas as pd
import torch.cuda
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
# from data_util import *
from transformers import Trainer, TrainingArguments
from transformers.optimization import AdamW
from torch.utils.data import Dataset, random_split
from sklearn.model_selection import train_test_split
from metrices import *

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-train_path', type=str, default=None)
    parser.add_argument('-train_batch', type=int, default=1)
    parser.add_argument('-eval_batch',type=int,default=1)
    parser.add_argument('-gpu', type=str, default='0')

    args = parser.parse_args()
    return args


# make sure GPT2 appends EOS in begin and end
def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
    return outputs


class NetflixDataset(Dataset):
    def __init__(self, txt_list, tokenizer, max_length, train_batch_size,eval_batch_size):
        train,eval = train_test_split(txt_list,train_size=0.7,test_size=0.3)
        eval,test = train_test_split(eval,train_size=0.5,test_size=0.5)
        self.length = len(txt_list)
        self.train = []
        self.eval = []
        self.test = []

        for i in range(0, len(train), train_batch_size):
            txt_batch = train[i:i + train_batch_size]
            encodings_batch = tokenizer.batch_encode_plus(txt_batch, truncation=True,
                                                          max_length=max_length, padding="max_length",
                                                          return_tensors='pt')
            self.train.append(encodings_batch)

        for i in range(0, len(eval), eval_batch_size):
            txt_batch = eval[i:i + eval_batch_size]
            encodings_batch = tokenizer.batch_encode_plus(txt_batch, truncation=True,
                                                          max_length=max_length, padding="max_length",
                                                          return_tensors='pt')
            self.eval.append(encodings_batch)

        for i in range(0, len(test), eval_batch_size):
            txt_batch = test[i:i + eval_batch_size]
            encodings_batch = tokenizer.batch_encode_plus(txt_batch, truncation=True,
                                                          max_length=max_length, padding="max_length",
                                                          return_tensors='pt')
            self.test.append(encodings_batch)




if __name__ == '__main__':
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    descriptions = pd.read_csv('netflix_titles.csv')['description'].values[:50]

    GPT2Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # set pad_token_id to unk_token_id -> be careful here as unk_token_id == eos_token_id == bos_token_id
    tokenizer.pad_token = tokenizer.unk_token
    model = GPT2LMHeadModel.from_pretrained('gpt2', return_dict=True).to(device)

    # texts = ['I am a random text!','You are a monster!']
    dataset = NetflixDataset(descriptions, tokenizer, max_length=1024, train_batch_size=args.train_batch,eval_batch_size=args.eval_batch)
    train_set,eval_set,test_set = dataset.train,dataset.eval,dataset.test
    # print(tokenizer.batch_decode(eval_set[0]['input_ids'],skip_special_tokens=True))
    # quit()
    # inputs = tokenizer.batch_encode_plus(texts,return_tensors='pt',truncation=True,max_length=1024,padding=True).to(device)
    optimizer = AdamW(
        [
            {'params': model.parameters(), 'lr': 1e-5},
        ])
    initial_bleu = -1
    # train
    epoch = 3
    for ep in range(epoch):
        model.train()
        for train_batch in train_set:
            train_batch = train_batch.to(device)
            optimizer.zero_grad()
            outputs = model(**train_batch, labels=train_batch['input_ids'])
            loss = outputs.loss
            logits = outputs.logits
            indices = torch.argmax(logits, dim=2)
            # print(tokenizer.batch_decode(indices, skip_special_tokens=True))
            # print(loss.item())
            loss.backward()
            optimizer.step()

        # evaluation
        torch.save(
            {'gpt2': model.state_dict(),
             },
            f'checkpoints/epoch-{ep}.pt')

        model.eval()
        references = []
        candidates = []
        for eval_batch in eval_set:
            references += tokenizer.batch_decode(eval_batch['input_ids'],skip_special_tokens=True)
            eval_batch = eval_batch.to(device)
            outputs = model(**eval_batch, labels=eval_batch['input_ids'])
            logits = outputs.logits
            indices = torch.argmax(logits, dim=2)
            # print(tokenizer.batch_decode(indices, skip_special_tokens=True))
            candidates += tokenizer.batch_decode(indices, skip_special_tokens=True)
            # sample_outputs = model.generate(eval_batch['input_ids'], do_sample=True, top_k=50,
            #                                 max_length=300, top_p=0.95, temperature=1.9, num_return_sequences=1)
            # print(sample_outputs)
            # quit()
        # print(references)
        # print(candidates)
        assert len(references) == len(candidates)
        current_bleu = bleu(references=references,hypothesis=candidates)




