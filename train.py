import argparse
import os

import pandas as pd
import torch.cuda
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
# from data_util import *
from transformers import Trainer, TrainingArguments
from transformers.optimization import AdamW
from torch.utils.data import Dataset, random_split
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-train_path', type=str,default=None)
    parser.add_argument('-batch_size',type=int,default=32)
    parser.add_argument('-gpu',type=str,default='0')

    args = parser.parse_args()
    return args
args = get_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '4'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
descriptions = pd.read_csv('netflix_titles.csv')['description'].values

# make sure GPT2 appends EOS in begin and end
def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
    return outputs

class NetflixDataset(Dataset):
    def __init__(self, txt_list, tokenizer, max_length,batch_size):
        self.length = len(txt_list)
        self.batchs = []

        for i in range(0,len(txt_list),batch_size):
            txt_batch = txt_list[i:i+batch_size]
            encodings_batch = tokenizer.batch_encode_plus(txt_batch, truncation=True,
                                       max_length=max_length, padding="max_length",return_tensors='pt')
            self.batchs.append(encodings_batch.to(device))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.batchs

GPT2Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# set pad_token_id to unk_token_id -> be careful here as unk_token_id == eos_token_id == bos_token_id
tokenizer.pad_token = tokenizer.unk_token
model = GPT2LMHeadModel.from_pretrained('gpt2', return_dict=True).to(device)

# texts = ['I am a random text!','You are a monster!']
dataset = NetflixDataset(descriptions, tokenizer, max_length=1024,batch_size=args.batch_size)

# inputs = tokenizer.batch_encode_plus(texts,return_tensors='pt',truncation=True,max_length=1024,padding=True).to(device)
optimizer = AdamW(
                [
                {'params':model.parameters(),'lr':1e-5},
                ])

epoch = 3
for ep in range(epoch):
    model.train()
    for batch in dataset.batchs:
        optimizer.zero_grad()
        outputs = model(**batch,labels=batch['input_ids'])
        loss = outputs.loss
        logits = outputs.logits
        indices = torch.argmax(logits, dim=2)
        print(tokenizer.batch_decode(indices))
        print(loss.item())
        loss.backward()
        optimizer.step()

quit()
generated = tokenizer("<|endoftext|>Ouyang and fish", return_tensors="pt").input_ids.to(device)

# sample_outputs = model.generate(generated, do_sample=True, top_k=2,
#                                 max_length=64, top_p=0.95, temperature=1.9, num_return_sequences=3)
#
# for output in sample_outputs:
#     print(output)
#     print(tokenizer.decode(output,skip_special_tokens=True))
# quit()

args = get_args()
train_path = args.train_path
test_path = args.test_path
train_dataset, test_dataset, data_collator = load_dataset(train_path, test_path, tokenizer)

training_args = TrainingArguments(
    output_dir="./gpt2-finetuned", #The output directory
    overwrite_output_dir=True, #overwrite the content of the output directory
    num_train_epochs=3, # number of training epochs
    per_device_train_batch_size=32, # batch size for training
    per_device_eval_batch_size=64,  # batch size for evaluation
    eval_steps = 400, # Number of update steps between two evaluations.
    save_steps=800, # after # steps model is saved
    warmup_steps=500,# number of warmup steps for learning rate scheduler
    prediction_loss_only=True,
    )


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)
trainer.train()


trainer.save_model()