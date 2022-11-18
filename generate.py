import argparse
import json
import os
import pandas as pd
import torch
from data_util import get_prompt_simple,get_prompt_input
from transformers import BartForConditionalGeneration, BartTokenizer
from tqdm import trange


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
    data_path = '../../AmazonKG_Mave_Merged/MAVE_filtered.csv'
    # data_path = '../MAVE_filtered.csv'

    df = pd.read_csv(data_path)[['title','description','attributes']]

    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(device)
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    all_attributes = []
    for attribute in df['attributes'].values:
        if not pd.isna(attribute):
            attributes = list(json.loads(attribute).keys())
            for a in attributes:
                if a not in all_attributes:
                    all_attributes.append(a)



    inputs= get_prompt_input(df,tokenizer.mask_token,all_attributes)

    model.load_state_dict(torch.load("checkpoints/bart_epoch-7.pt"))
    model.eval()


    result = {}



    titles = list(inputs.keys())
    for i in trange(len(titles)):
        title = titles[i]
        prompts = inputs[title]
        result[title] = {}
        for attribute in prompts.keys():
            result[title][attribute] = []
            prompt = prompts[attribute]
            text = prompt

            x = tokenizer.batch_encode_plus(text, truncation=True,
                                                    max_length=128, padding="longest",
                                                    return_tensors='pt').to(device)

            outputs = model.generate(
                input_ids=x.input_ids,
                attention_mask=x.attention_mask,
                max_length=10,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.pad_token_id,
                num_beams=28,
                early_stopping=True
            )

            generate_sentences = tokenizer.batch_decode(outputs,skip_special_tokens=True)
            print(text)
            print(generate_sentences)
            result[title][attribute] += generate_sentences
            quit()
    df = pd.DataFrame()
    titles = []
    attributes = []
    for title in result.keys():
        titles.append(title)
        attributes.append(result[title])
    df['title'] = titles
    df['attributes'] = attributes
    df.to_csv('generate_attributes.csv')