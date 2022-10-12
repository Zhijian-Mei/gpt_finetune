import argparse
import os

import pandas as pd
import torch.cuda
from tqdm import trange
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
# from data_util import *
from transformers import Trainer, TrainingArguments
from torch.optim import AdamW
# from transformers.optimization import AdamW
from torch.utils.data import Dataset, random_split
from sklearn.model_selection import train_test_split
from metrices import *
from openprompt import PromptForGeneration
from openprompt.prompts import ManualTemplate

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
    def __init__(self, df,  tokenizer, max_length, train_batch_size,eval_batch_size,template):
        train,eval = train_test_split(df,train_size=0.7,test_size=0.3)
        eval,test = train_test_split(eval,train_size=0.5,test_size=0.5)
        self.train_x = []
        self.train_y = []
        self.eval_x = []
        self.eval_y = []
        self.test_x = []
        self.test_y = []


        train_x = []
        train_y = []
        for type,des in train:
            input = f'{des}; {template}'
            train_x.append(input)
            refactor_template = template.replace(tokenizer.unk_token,type)
            output = f'{des}; {refactor_template}'
            train_y.append(output)
            # print(input)
            # print(output)
            # quit()

        for i in range(0, len(train_x), train_batch_size):
            train_batch_x = train_x[i:i + train_batch_size]
            encodings_batch_x = tokenizer.batch_encode_plus(train_batch_x, truncation=True,
                                                          max_length=max_length, padding="longest",
                                                          return_tensors='pt')
            train_batch_y = train_y[i:i + train_batch_size]
            encodings_batch_y = tokenizer.batch_encode_plus(train_batch_y, truncation=True,
                                                          max_length=max_length, padding="longest",
                                                          return_tensors='pt')
            self.train_x.append(encodings_batch_x)
            self.train_y.append(encodings_batch_y)


        eval_x = []
        eval_y = []

        for type,des in eval:
            input = f'{des}; {template}'
            eval_x.append(input)
            refactor_template = template.replace(tokenizer.unk_token,type)
            output = f'{des}; {refactor_template}'
            eval_y.append(output)

        for i in range(0, len(eval), eval_batch_size):
            eval_batch_x = eval_x[i:i + eval_batch_size]
            encodings_batch_x = tokenizer.batch_encode_plus(eval_batch_x, truncation=True,
                                                          max_length=max_length, padding="longest",
                                                          return_tensors='pt')
            self.eval_x.append(encodings_batch_x)
            self.eval_y.append(eval_y[i:i + eval_batch_size])

        test_x = []
        test_y = []

        for type,des in test:
            input = f'{des}; {template}'

            test_x.append(input)
            refactor_template = template.replace(tokenizer.unk_token,type)
            output = f'{des}; {refactor_template}'
            test_y.append(output)


        for i in range(0, len(test), eval_batch_size):
            test_batch_x = test_x[i:i + eval_batch_size]

            encodings_batch_x = tokenizer.batch_encode_plus(test_batch_x, truncation=True,
                                                          max_length=max_length, padding="longest",
                                                          return_tensors='pt')
            self.test_x.append(encodings_batch_x)
            self.test_y.append(test_y[i:i + eval_batch_size])


if __name__ == '__main__':
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    df = pd.read_csv('netflix_titles.csv')[['type','description']].values[:50]

    # GPT2Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # set pad_token_id to unk_token_id -> be careful here as unk_token_id == eos_token_id == bos_token_id
    tokenizer.pad_token = tokenizer.eos_token


    model = GPT2LMHeadModel.from_pretrained('gpt2', return_dict=True).to(device)

    template = f'The type of this Television work is {tokenizer.unk_token}'

    dataset = NetflixDataset(df, tokenizer, max_length=1024, train_batch_size=args.train_batch,eval_batch_size=args.eval_batch,template=template)
    train_x,train_y,eval_x,eval_y,test_x,test_y = dataset.train_x,dataset.train_y,dataset.eval_x,dataset.eval_y,dataset.test_x,dataset.test_y
    assert len(train_x) == len(train_y)
    assert len(eval_x) == len(eval_y)
    assert len(test_x) == len(test_y)
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
        for i in trange(len(train_x)):
            x = train_x[i]
            y = train_y[i]
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs = model(**x, labels=y['input_ids'])
            loss = outputs.loss
            # logits = outputs.logits
            # indices = torch.argmax(logits, dim=2)
            # print(tokenizer.batch_decode(indices, skip_special_tokens=True))
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f'Average loss epoch {ep} = {total_loss/len(train_x)}')
        # quit()
        # evaluation
        torch.save(
            {'gpt2': model.state_dict(),
             },
            f'checkpoints/epoch-{ep}.pt')

        model.eval()
        references = []
        candidates = []
        for i in trange(len(eval_x)):
            x = eval_x[i]
            y = eval_y[i]
            print(y)
            quit()
            x = x.to(device)
            references += y
            eval_batch = eval_batch.to(device)
            # outputs = model(**eval_batch, labels=eval_batch['input_ids'])
            # logits = outputs.logits
            # indices = torch.argmax(logits, dim=2)
            # candidates += tokenizer.batch_decode(indices, skip_special_tokens=True)
            # print(tokenizer.batch_decode(indices, skip_special_tokens=True))
            sample_outputs = model.generate(
                input_ids=eval_batch.input_ids,
                attention_mask=eval_batch.attention_mask,
                max_length =128,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.eos_token_id,
                num_beams=2,
                early_stopping=True
            )


            candidates += tokenizer.batch_decode(sample_outputs,skip_special_tokens=True)
            # print(tokenizer.batch_decode(sample_outputs,skip_special_tokens=True))
            # quit()
        print(references)
        print(candidates)

        assert len(references) == len(candidates)
        current_bleu = bleu(references=references,hypothesis=candidates)
        print(current_bleu)



