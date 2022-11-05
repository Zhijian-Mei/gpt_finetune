import argparse
import os

import torch
from tqdm import trange
from transformers import BartForConditionalGeneration,BartTokenizer
from data_util import *
from train import MyDataset
from metrices import bleu

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-train_path', type=str, default=None)
    parser.add_argument('-train_batch', type=int, default=2)
    parser.add_argument('-eval_batch',type=int,default=2)
    parser.add_argument('-gpu', type=str, default='0')
    parser.add_argument('-epoch', type=str,default='3')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # data_path = '../../AmazonKG_Mave_Merged/MAVE_filtered.csv'
    data_path = '../MAVE_filtered.csv'


    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    eval_batch_size = args.eval_batch

    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(device)
    checkpoint = torch.load(f'checkpoints/bart_epoch-{args.epoch}.pt', map_location='cuda')
    model.load_state_dict(checkpoint)
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    df = pd.read_csv(data_path)[['title', 'description', 'attributes']]

    inputs, outputs = get_prompt_simple(df, tokenizer.mask_token)
    df = pd.DataFrame()
    df['inputs'] = inputs
    df['outputs'] = outputs

    _, _, test = get_data(df,random_seed=48)

    testDataset = MyDataset(test,tokenizer,batch_size=eval_batch_size)


    model.eval()
    references = []
    candidates = []

    for i in trange(len(testDataset)):
        batch = testDataset[i]
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
            max_length=128,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.pad_token_id,
            num_beams=2,
            early_stopping=True
        )
        candidates += tokenizer.batch_decode(sample_outputs, skip_special_tokens=True)

    assert len(references) == len(candidates)
    bleu = bleu(references=references, hypothesis=candidates)

    print(bleu)