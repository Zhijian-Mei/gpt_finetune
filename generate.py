import torch
from transformers import BartForConditionalGeneration, BartTokenizer



text = ["Blackmailed by his company's CEO, a low-level employee finds himself forced to spy on the boss's rival and former mentor.; The type of this television work is ",
'As big city life buzzes around them, lonely souls discover surprising sources of connection and companionship in three tales of love, loss and longing.; The type of this television work is'

        ]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(device)
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")


model.load_state_dict(torch.load("checkpoints/epoch-2.pt")['bart'])
model.eval()
x = tokenizer.batch_encode_plus(text, truncation=True,
                                                max_length=64, padding="longest",
                                                return_tensors='pt').to(device)
print(x)
sample_outputs = model.generate(
    input_ids=x.input_ids,
    attention_mask=x.attention_mask,
    max_length=128,
    no_repeat_ngram_size=2,
    pad_token_id=tokenizer.pad_token_id,
    num_beams=2,
    early_stopping=True
)

print(sample_outputs)
print(tokenizer.batch_decode(sample_outputs,skip_special_tokens=True))