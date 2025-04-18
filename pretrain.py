#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaTokenizer, RobertaModel
import torch
from tqdm.auto import tqdm

print(torch.cuda.is_available())

paths = [str(x) for x in Path('/processing/l.cai/text_radiology_NEW_smalldataset_only_radiology').glob('**/*.txt')]

#tokenizer = ByteLevelBPETokenizer()

#tokenizer.train(files=paths[:], vocab_size=30_522, min_frequency=2, special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'])


#tokenizer.save_model('/projects/gan-mri-data/mul-data/NLP/Lishan')

# initialize the tokenizer using the tokenizer we initialized and saved to file
tokenizer = RobertaTokenizer.from_pretrained('/data/groups/beets-tan/l.cai/rectal_nlp/Lishan_nlp', max_len=512)


def mlm(tensor):
  rand = torch.rand(tensor.shape)
  mask_arr = (rand < 0.15) * (tensor > 2)
  for i in range(tensor.shape[0]):
    selection = torch.flatten(mask_arr[i].nonzero()).tolist()
    tensor[i, selection] = 4
  return tensor
  

paths = [str(x) for x in Path('/processing/l.cai/text_radiology_NEW_smalldataset').glob('**/*.txt')]
print(paths[:])



input_ids = []
mask = []
labels = []

for path in tqdm(paths[:]):
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
  sample = tokenizer(lines, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
  labels.append(sample.input_ids)
  mask.append(sample.attention_mask)
  input_ids.append(mlm(sample.input_ids.detach().clone()))

input_ids = torch.cat(input_ids)
mask = torch.cat(mask)
labels = torch.cat(labels)

encodings = {
    'input_ids': input_ids,
    'attention_mask': mask,
    'labels': labels
}

class Dataset(torch.utils.data.Dataset):
  def __init__(self, encodings):
    self.encodings = encodings

  def __len__(self):
    return self.encodings['input_ids'].shape[0]

  def __getitem__(self, i):
    return {key: tensor[i] for key, tensor in self.encodings.items()}
    
dataset = Dataset(encodings)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)




from transformers import RobertaConfig
config = RobertaConfig(
    vocab_size=tokenizer.vocab_size,
    max_position_embeddings =514,
    hidden_size=768,
    num_attention_heads=12,
    num_hidden_layers=12,
    type_vocab_size=1
)

from transformers import RobertaForMaskedLM, AdamW
model = RobertaForMaskedLM(config)
#model.load_state_dict(torch.load('/data/groups/beets-tan/l.cai/rectal_nlp/Lishan_nlp/pytorch_model.bin'))
#model = model = torch.load('/data/groups/beets-tan/l.cai/rectal_nlp/Lishan_nlp/pytorch_model.bin')
#model = RobertaModel.from_pretrained('/data/groups/beets-tan/l.cai/rectal_nlp/Lishan_nlp/')
#model = model.load('/data/groups/beets-tan/l.cai/rectal_nlp/Lishan_nlp/pytorch_model.bin')

## this is to load the pretrained weights
model = RobertaForMaskedLM.from_pretrained('/data/groups/beets-tan/l.cai/rectal_nlp/Lishan_nlp/')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.train()
optim =  AdamW(model.parameters(), lr=1e-4)

from torch.utils import tensorboard
writer = torch.utils.tensorboard.SummaryWriter()
epochs = 5
step = 0
def train_model(epochs):
  for epoch in range(epochs):
    loop = tqdm(dataloader, leave=True)
    for batch in loop:
      optim.zero_grad()
      input_ids = batch['input_ids'].to(device)
      mask = batch['attention_mask'].to(device)
      labels = batch['labels'].to(device)
      outputs = model(input_ids, attention_mask=mask, labels=labels)
      loss = outputs.loss
      writer.add_scalar('Loss/train', loss, step)
      loss.backward()
      optim.step()
      
      loop.set_description(f'Epoch: {epoch}')
      loop.set_postfix(loss=loss.item())

train_model(epochs)
writer.flush()
model.save_pretrained('/data/groups/beets-tan/l.cai/rectal_nlp/trained_models_5_only_radiology')