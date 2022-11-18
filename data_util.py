import json
import random

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import TextDataset, DataCollatorForLanguageModeling

def load_dataset(train_path, test_path, tokenizer):
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_path,
        block_size=128)

    test_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=test_path,
        block_size=128)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset, test_dataset, data_collator

def get_data(df,random_seed=48):
    train, eval = train_test_split(df, train_size=0.7, test_size=0.3, random_state=random_seed)
    eval, test = train_test_split(eval, train_size=0.5, test_size=0.5, random_state=random_seed)

    return train, eval, test

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
                temp_output = prompt
                inputs.append(temp_input)
                outputs.append(temp_output)
            continue
        inputs.append(input_base)
        outputs.append(input_base)

    return inputs,outputs

def get_prompt_input(df,mask_token,attributes):
    inputs = {}
    titles = df['title']
    descriptions = df['description']
    for i in range(len(df)):
        if pd.isna(titles[i]):
            continue
        inputs[titles[i]] = {}
        input_base = ''
        input_base += titles[i] + '; '
        if not pd.isna(descriptions[i]):
            input_base += descriptions[i][2:-2] + '; '
        for key in attributes:
            if key not in inputs[titles[i]].keys():
                inputs[titles[i]][key] = []
                masked_prompt = f'the {key} of the item is {mask_token}'
                temp_input = input_base + masked_prompt
                inputs[titles[i]][key].append(temp_input)

    return inputs

