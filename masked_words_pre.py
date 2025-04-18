import os
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaTokenizer
import torch
from tqdm.auto import tqdm
from sklearn import metrics
from transformers import BertTokenizer, BertModel, BertConfig, RobertaTokenizer, RobertaTokenizerFast, RobertaForMaskedLM
from transformers import AutoTokenizer, AutoModelForMaskedLM



class Dataset(torch.utils.data.Dataset):
  def __init__(self, encodings):
    self.encodings = encodings

  def __len__(self):
    return self.encodings['input_ids'].shape[0]

  def __getitem__(self, i):
    return {key: tensor[i] for key, tensor in self.encodings.items()}




paths = [str(x) for x in Path('/data/groups/beets-tan/l.cai/rectal_nlp/Test').glob('**/*.txt')]

tokenizer_small = RobertaTokenizer.from_pretrained('/data/groups/beets-tan/l.cai/rectal_nlp/Lishan_nlp', max_len=512)
tokenizer_big = RobertaTokenizer.from_pretrained('/data/groups/beets-tan/l.cai/rectal_nlp/trained_models', max_len=512)
#tokenizer_raw = RobertaTokenizer.from_pretrained('pdelobelle/robbert-v2-dutch-base')#AutoTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base")
tokenizer_raw = RobertaTokenizer.from_pretrained("CLTL/MedRoBERTa.nl")
#model = AutoModelForMaskedLM.from_pretrained("CLTL/MedRoBERTa.nl")
#model_raw = AutoModelForMaskedLM.from_pretrained("pdelobelle/robbert-v2-dutch-base")

def mlm(tensor):
  rand = torch.rand(tensor.shape)
  mask_arr = (rand < 0.15) * (tensor > 2)
  for i in range(tensor.shape[0]):
    selection = torch.flatten(mask_arr[i].nonzero()).tolist()
    tensor[i, selection] = 4
  return tensor

input_ids_small = []
mask_small = []
labels_small = []

input_ids_big = []
mask_big = []
labels_big = []

input_ids_raw = []
mask_raw = []
labels_raw = []


for path in tqdm(paths[:]):
  print(paths)
  print(path)
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
  sample_small = tokenizer_small(lines, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
  sample_big = tokenizer_big(lines, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
  sample_raw = tokenizer_raw(lines, max_length=512, padding='max_length', truncation=True, return_tensors='pt')

  print('tokenizer',sample_small['input_ids'].shape, sample_big['input_ids'].shape, sample_raw['input_ids'].shape)

  labels_small.append(sample_small.input_ids)
  mask_small.append(sample_small.attention_mask)
  input_ids_small.append(mlm(sample_small.input_ids.detach().clone()))

  labels_big.append(sample_big.input_ids)
  mask_big.append(sample_big.attention_mask)
  input_ids_big.append(mlm(sample_big.input_ids.detach().clone()))

  labels_raw.append(sample_raw.input_ids)
  mask_raw.append(sample_raw.attention_mask)
  input_ids_raw.append(mlm(sample_raw.input_ids.detach().clone()))



input_ids_small = torch.cat(input_ids_small)
mask_small = torch.cat(mask_small)
labels_small = torch.cat(labels_small)

input_ids_big = torch.cat(input_ids_big)
mask_big = torch.cat(mask_big)
labels_big = torch.cat(labels_big)

input_ids_raw = torch.cat(input_ids_raw)
mask_raw = torch.cat(mask_raw)
labels_raw = torch.cat(labels_raw)


input_ids_big_666= torch.where(input_ids_small == 4, input_ids_small, labels_big)
input_ids_raw_consis = torch.where(input_ids_small == 4, input_ids_small, labels_raw)

encodings_small = {
    'input_ids': input_ids_small[:,:],
    'attention_mask': mask_small[:,:],
    'labels': labels_small[:,:]
}

encodings_big = {
    'input_ids': input_ids_big_666[:,:],
    'attention_mask': mask_big[:,:],
    'labels': labels_big[:,:]
}


encodings_raw = {
    'input_ids': input_ids_raw_consis[:,:],
    'attention_mask': mask_raw[:,:],
    'labels': labels_raw[:,:]
}


dataset_small = Dataset(encodings_small)
dataloader_small = torch.utils.data.DataLoader(dataset_small, batch_size=64, shuffle=False)

dataset_big = Dataset(encodings_big)
dataloader_big = torch.utils.data.DataLoader(dataset_big, batch_size=64, shuffle=False)


dataset_raw = Dataset(encodings_raw)
dataloader_raw = torch.utils.data.DataLoader(dataset_raw, batch_size=64, shuffle=False)

print(len(dataset_small), len(dataset_big), len(dataset_raw))
device = 'cuda'
#################################################################################################################################
### radiobert general dutch
model_raw = RobertaForMaskedLM.from_pretrained('CLTL/MedRoBERTa.nl')#("pdelobelle/robbert-v2-dutch-base")
#model_small=RobertaForMaskedLM.from_pretrained('/data/groups/beets-tan/l.cai/rectal_nlp/Lishan_nlp', return_dict=False)
model_raw.to('cuda')
model_raw.eval()

## how to select top1, top5 and top10??
XX_small=[]
YY_small=[]
TOP5_small=[]
TOP10_small=[]
output_viz_small=[]
input_viz_small=[]
labels_viz_small=[]
with torch.no_grad():
  loop = tqdm(dataloader_raw, leave=True)
  print(len(loop))
  for batch in loop:
    input_ids6 = batch['input_ids'].to(device)
    mask6 = batch['attention_mask'].to(device)
    labels6 = batch['labels'].to(device)
    #print(labels6)
    outputs = model_raw(input_ids6, attention_mask=mask6)
    outputs123=torch.flatten(outputs[0],start_dim=-1)  #!!!
    #print(outputs)
    #print(outputs123)
    b123=torch.reshape(outputs123, (outputs123.shape[0]*outputs123.shape[1],outputs123.shape[2]))
    #print(outputs123.shape[0])
    #print(outputs)
    y=torch.argmax(outputs[0], -1)
    x=input_ids6
    z=labels6
    #print(y)
    y123=torch.flatten(y)  #!!!
    x123=torch.flatten(x)  #!!!
    z123=torch.flatten(z)  #!!!
    #print(outputs123.shape)
    #print(b123.shape)
    b5=torch.topk(b123, 5)[1]
    print('top5', b5)
    #print(b)
    #ttt=b[0]
    #b123=torch.flatten(ttt,start_dim=1)  #!!!
    #print(ttt)
    #print("b123:",b123)
    T5=[]
    for i in range(512*outputs123.shape[0]):
      if z123[i] in b5[i]:
        r1=z123[i].tolist()
        T5.append(r1)
      else:
        r2=b5[i][0].tolist()
        T5.append(r2)
    t5=torch.LongTensor(T5).to(device)

    b10=torch.topk(b123, 10)[1]
    #print(b)
    #ttt=b[0]
    #b123=torch.flatten(ttt,start_dim=1)  #!!!
    #print(ttt)
    #print("b123:",b123)
    T10=[]
    for i in range(512*outputs123.shape[0]):
      if z123[i] in b10[i]:
        r1=z123[i].tolist()
        T10.append(r1)
      else:
        r2=b10[i][0].tolist()
        T10.append(r2)

    t10=torch.LongTensor(T10).to(device)
    #print(x123.shape)
    #print(t5.shape)
    p=torch.where(x123 == 4, y123, 0)
    g=torch.where(x123 == 4, z123, 0)
    t_5=torch.where(x123 == 4, t5, 0)
    t_10=torch.where(x123 == 4, t10, 0)

    pp= p.tolist()
    #print(pp)
    gg= g.tolist()
    tt_5= t_5.tolist()
    tt_10= t_10.tolist()
    X = [i for i in pp if i != 0]
    Y = [i for i in gg if i != 0]
    T_5 = [i for i in tt_5 if i != 0]
    T_10 = [i for i in tt_10 if i != 0]
    XX_small.append(X)
    YY_small.append(Y)
    TOP5_small.append(T_5)
    TOP10_small.append(T_10)

    # viz

    out_small=torch.argmax(outputs[0],-1)
    LX=[]
    LY=[]
    LG=[]
    for i in range(outputs123.shape[0]):
      lx=input_ids6[i]
      lx1=lx.tolist()
      #lx666=tokenizer.decode(lx1)

      ly=out_small[i]
      ly1=ly.tolist()
      #ly666=tokenizer.decode(ly1)

      lg=labels6[i]
      lg1=lg.tolist()
      #lg666=tokenizer.decode(lg1)
      LX.append(lx1)
      LY.append(ly1)
      LG.append(lg1)

    input_viz_small.append(LX)
    output_viz_small.append(LY)
    labels_viz_small.append(LG)
#################################################################################################################################
### radiobert_SmallDataset_epoch10
outputs_small=sum(XX_small, [])
#print(outputs12)
gt_small=sum(YY_small, [])
#print(gt12)
top5_small=sum(TOP5_small, [])
#print(top5)
top10_small=sum(TOP10_small, [])
#print(top10)

print("output_small        len:", len(outputs_small))
print("groundtruth_small   len:", len(gt_small))
print("top5_small          len:", len(top5_small))
print("top10_small         len:", len(top10_small))

if len(outputs_small) == len(gt_small) == len(top5_small) == len(top10_small):
  print("Num is equal!       len:", len(outputs_small))

else:
  print("Error!!!")

accuracy_small_top1 = metrics.accuracy_score(outputs_small, gt_small)
accuracy_small_top5 = metrics.accuracy_score(top5_small, gt_small)
accuracy_small_top10 = metrics.accuracy_score(top10_small, gt_small)
print("model           name: radiobert_SmallDataset_e10")
print("total            num:", len(outputs_small))
print("correct          num:", int(accuracy_small_top1*len(outputs_small)))
print("------------------------")
print("accuracy_small_top1 :", "%.5f" % accuracy_small_top1)
print("accuracy_small_top5 :", "%.5f" % accuracy_small_top5)
print("accuracy_small_top10:", "%.5f" % accuracy_small_top10)

input_viz_small_c=sum(input_viz_small, [])
output_viz_small_c=sum(output_viz_small, [])
labels_viz_small_c=sum(labels_viz_small, [])
num=[]











device = 'cuda'
#################################################################################################################################
### radiobert_SmallDataset_epoch10

model_small=RobertaForMaskedLM.from_pretrained('/data/groups/beets-tan/l.cai/rectal_nlp/Lishan_nlp', return_dict=False)
model_small.to('cuda')
model_small.eval()

## how to select top1, top5 and top10??
XX_small=[]
YY_small=[]
TOP5_small=[]
TOP10_small=[]
output_viz_small=[]
input_viz_small=[]
labels_viz_small=[]
with torch.no_grad():
  loop = tqdm(dataloader_small, leave=True)
  for batch in loop:
    input_ids6 = batch['input_ids'].to(device)
    mask6 = batch['attention_mask'].to(device)
    labels6 = batch['labels'].to(device)
    #print(labels6)
    outputs = model_small(input_ids6, attention_mask=mask6)
    outputs123=torch.flatten(outputs[0],start_dim=-1)  #!!!
    #print(outputs)
    #print(outputs123)
    b123=torch.reshape(outputs123, (outputs123.shape[0]*outputs123.shape[1],outputs123.shape[2]))
    #print(outputs123.shape[0])
    #print(outputs)
    y=torch.argmax(outputs[0], -1)
    x=input_ids6
    z=labels6
    #print(y)
    y123=torch.flatten(y)  #!!!
    x123=torch.flatten(x)  #!!!
    z123=torch.flatten(z)  #!!!
    #print(outputs123.shape)
    #print(b123.shape)
    b5=torch.topk(b123, 5)[1]
    #print(b)
    #ttt=b[0]
    #b123=torch.flatten(ttt,start_dim=1)  #!!!
    #print(ttt)
    #print("b123:",b123)
    T5=[]
    for i in range(512*outputs123.shape[0]):
      if z123[i] in b5[i]:
        r1=z123[i].tolist()
        T5.append(r1)
      else:
        r2=b5[i][0].tolist()
        T5.append(r2)
    t5=torch.LongTensor(T5).to(device)

    b10=torch.topk(b123, 10)[1]
    #print(b)
    #ttt=b[0]
    #b123=torch.flatten(ttt,start_dim=1)  #!!!
    #print(ttt)
    #print("b123:",b123)
    T10=[]
    for i in range(512*outputs123.shape[0]):
      if z123[i] in b10[i]:
        r1=z123[i].tolist()
        T10.append(r1)
      else:
        r2=b10[i][0].tolist()
        T10.append(r2)

    t10=torch.LongTensor(T10).to(device)
    #print(x123.shape)
    #print(t5.shape)
    p=torch.where(x123 == 4, y123, 0)
    g=torch.where(x123 == 4, z123, 0)
    t_5=torch.where(x123 == 4, t5, 0)
    t_10=torch.where(x123 == 4, t10, 0)

    pp= p.tolist()
    #print(pp)
    gg= g.tolist()
    tt_5= t_5.tolist()
    tt_10= t_10.tolist()
    X = [i for i in pp if i != 0]
    Y = [i for i in gg if i != 0]
    T_5 = [i for i in tt_5 if i != 0]
    T_10 = [i for i in tt_10 if i != 0]
    XX_small.append(X)
    YY_small.append(Y)
    TOP5_small.append(T_5)
    TOP10_small.append(T_10)

    # viz

    out_small=torch.argmax(outputs[0],-1)
    LX=[]
    LY=[]
    LG=[]
    for i in range(outputs123.shape[0]):
      lx=input_ids6[i]
      lx1=lx.tolist()
      #lx666=tokenizer.decode(lx1)

      ly=out_small[i]
      ly1=ly.tolist()
      #ly666=tokenizer.decode(ly1)

      lg=labels6[i]
      lg1=lg.tolist()
      #lg666=tokenizer.decode(lg1)
      LX.append(lx1)
      LY.append(ly1)
      LG.append(lg1)

    input_viz_small.append(LX)
    output_viz_small.append(LY)
    labels_viz_small.append(LG)
#################################################################################################################################
### radiobert_SmallDataset_epoch10
outputs_small=sum(XX_small, [])
#print(outputs12)
gt_small=sum(YY_small, [])
#print(gt12)
top5_small=sum(TOP5_small, [])
#print(top5)
top10_small=sum(TOP10_small, [])
#print(top10)

print("output_small        len:", len(outputs_small))
print("groundtruth_small   len:", len(gt_small))
print("top5_small          len:", len(top5_small))
print("top10_small         len:", len(top10_small))

if len(outputs_small) == len(gt_small) == len(top5_small) == len(top10_small):
  print("Num is equal!       len:", len(outputs_small))

else:
  print("Error!!!")

accuracy_small_top1 = metrics.accuracy_score(outputs_small, gt_small)
accuracy_small_top5 = metrics.accuracy_score(top5_small, gt_small)
accuracy_small_top10 = metrics.accuracy_score(top10_small, gt_small)
print("model           name: radiobert_SmallDataset_e10")
print("total            num:", len(outputs_small))
print("correct          num:", int(accuracy_small_top1*len(outputs_small)))
print("------------------------")
print("accuracy_small_top1 :", "%.5f" % accuracy_small_top1)
print("accuracy_small_top5 :", "%.5f" % accuracy_small_top5)
print("accuracy_small_top10:", "%.5f" % accuracy_small_top10)

input_viz_small_c=sum(input_viz_small, [])
output_viz_small_c=sum(output_viz_small, [])
labels_viz_small_c=sum(labels_viz_small, [])
num=[]
# with open('viz_small_new.txt', 'w') as f:
#   for i in range(len(input_viz_small_c)):
#     lx=input_viz_small_c[i]
#     lx6=tokenizer_small.decode(lx)
#     lx666=lx6.replace("<pad>","")

#     ly=output_viz_small_c[i]
#     ly6=tokenizer_small.decode(ly)
#     ly666=ly6.replace("<pad>","")

#     lg=labels_viz_small_c[i]
#     lg6=tokenizer_small.decode(lg)
#     lg666=lg6.replace("<pad>","")
#     num.append(i)
#     f.write("case: ")
#     f.write(str(len(num)))
#     f.write("!!!---------------------------------")
#     f.write('\n')
#     f.write("[input             (masked)]: ")
#     f.write(lx666)
#     f.write('\n')
#     f.write("[output_samll  (pridiction)]: ")
#     f.write(ly666)
#     f.write('\n')
#     f.write("[label       (ground truth)]: ")
#     f.write(lg666)
#     f.write('\n')
    # print("case",len(num),"!!!---------------------------------")
    # print("[input       (masked)]:", lx666)
    # print("[output  (pridiction)]:", ly666)
    # print("[label (ground truth)]:", lg666)
#################################################################################################################################





















#################################################################################################################################
### radiobert_BigDataset_epoch10
model_big=RobertaForMaskedLM.from_pretrained('/data/groups/beets-tan/l.cai/rectal_nlp/trained_models', return_dict=False)
model_big.to(device)
model_big.eval()

XX_big=[]
YY_big=[]
TOP5_big=[]
TOP10_big=[]
output_viz_big=[]
input_viz_big=[]
labels_viz_big=[]
with torch.no_grad():
  loop = tqdm(dataloader_big, leave=True)
  for batch in loop:
    input_ids6 = batch['input_ids'].to(device)
    mask6 = batch['attention_mask'].to(device)
    labels6 = batch['labels'].to(device)
    #print(labels6)
    outputs = model_big(input_ids6, attention_mask=mask6)
    outputs123=torch.flatten(outputs[0],start_dim=-1)  #!!!
    #print(outputs)
    #print(outputs123)
    b123=torch.reshape(outputs123, (outputs123.shape[0]*outputs123.shape[1],outputs123.shape[2]))
    #print(outputs123.shape[0])

    #print(outputs)
    y=torch.argmax(outputs[0], -1)
    x=input_ids6
    z=labels6
    #print(y)
    y123=torch.flatten(y)  #!!!
    x123=torch.flatten(x)  #!!!
    z123=torch.flatten(z)  #!!!
    #print(outputs123.shape)
    #print(b123.shape)
    b5=torch.topk(b123, 5)[1]
    #print(b)
    #ttt=b[0]
    #b123=torch.flatten(ttt,start_dim=1)  #!!!
    #print(ttt)
    #print("b123:",b123)
    T5=[]
    for i in range(512*outputs123.shape[0]):
      if z123[i] in b5[i]:
        r1=z123[i].tolist()
        T5.append(r1)
      else:
        r2=b5[i][0].tolist()
        T5.append(r2)
    t5=torch.LongTensor(T5).to(device)

    b10=torch.topk(b123, 10)[1]
    #print(b)
    #ttt=b[0]
    #b123=torch.flatten(ttt,start_dim=1)  #!!!
    #print(ttt)
    #print("b123:",b123)
    T10=[]
    for i in range(512*outputs123.shape[0]):
      if z123[i] in b10[i]:
        r1=z123[i].tolist()
        T10.append(r1)
      else:
        r2=b10[i][0].tolist()
        T10.append(r2)

    t10=torch.LongTensor(T10).to(device)


    #print(x123.shape)
    #print(t5.shape)
    p=torch.where(x123 == 4, y123, 0)
    g=torch.where(x123 == 4, z123, 0)
    t_5=torch.where(x123 == 4, t5, 0)
    t_10=torch.where(x123 == 4, t10, 0)

    pp= p.tolist()
    #print(pp)
    gg= g.tolist()
    tt_5= t_5.tolist()
    tt_10= t_10.tolist()
    X = [i for i in pp if i != 0]
    Y = [i for i in gg if i != 0]
    T_5 = [i for i in tt_5 if i != 0]
    T_10 = [i for i in tt_10 if i != 0]
    XX_big.append(X)
    YY_big.append(Y)
    TOP5_big.append(T_5)
    TOP10_big.append(T_10)

    # viz

    out_big=torch.argmax(outputs[0],-1)
    LX=[]
    LY=[]
    LG=[]
    for i in range(outputs123.shape[0]):
      lx=input_ids6[i]
      lx1=lx.tolist()
      #lx666=tokenizer.decode(lx1)

      ly=out_big[i]
      ly1=ly.tolist()
      #ly666=tokenizer.decode(ly1)

      lg=labels6[i]
      lg1=lg.tolist()
      #lg666=tokenizer.decode(lg1)
      LX.append(lx1)
      LY.append(ly1)
      LG.append(lg1)

    input_viz_big.append(LX)
    output_viz_big.append(LY)
    labels_viz_big.append(LG)
#################################################################################################################################
### radiobert_BigDataset_epoch10
outputs_big=sum(XX_big, [])
#print(outputs12)
gt_big=sum(YY_big, [])
#print(gt12)
top5_big=sum(TOP5_big, [])
#print(top5)
top10_big=sum(TOP10_big, [])
#print(top10)

print("output_big          len:", len(outputs_big))
print("groundtruth_big     len:", len(gt_big))
print("top5_big            len:", len(top5_big))
print("top10_big           len:", len(top10_big))

if len(outputs_big) == len(gt_big) == len(top5_big) == len(top10_big):
  print("Num is equal!       len:", len(outputs_big))

else:
  print("Error!!!")

accuracy_big_top1 = metrics.accuracy_score(outputs_big, gt_big)
accuracy_big_top5 = metrics.accuracy_score(top5_big, gt_big)
accuracy_big_top10 = metrics.accuracy_score(top10_big, gt_big)
print("model           name: radiobert_BigDataset_e10")
print("total            num:", len(outputs_big))
print("correct          num:", int(accuracy_big_top1*len(outputs_big)))
print("------------------------")
print("accuracy_big_top1   :", "%.5f" % accuracy_big_top1)
print("accuracy_big_top5   :", "%.5f" % accuracy_big_top5)
print("accuracy_big_top10  :", "%.5f" % accuracy_big_top10)

input_viz_big_c=sum(input_viz_big, [])
output_viz_big_c=sum(output_viz_big, [])
labels_viz_big_c=sum(labels_viz_big, [])

num=[]
# with open('viz_small and big_newnewnew.txt', 'w') as f:
#   for i in range(len(input_viz_big_c)):
#     lx=input_viz_big_c[i]
#     #lx1=lx.tolist()
#     lx6=tokenizer_big.decode(lx)
#     lx666=lx6.replace("<pad>","")

#     ly=output_viz_big_c[i]
#     #ly1=ly.tolist()
#     ly6=tokenizer_big.decode(ly)
#     ly666=ly6.replace("<pad>","")

#     lys=output_viz_small_c[i]
#     #ly1=ly.tolist()
#     lys6=tokenizer_small.decode(lys)
#     lys666=lys6.replace("<pad>","")

#     lg=labels_viz_big_c[i]
#     #lg1=lg.tolist()
#     lg6=tokenizer_big.decode(lg)
#     lg666=lg6.replace("<pad>","")
#     num.append(i)
#     f.write("case: ")
#     f.write(str(len(num)))
#     f.write("!!!---------------------------------")
#     f.write('\n')
#     f.write("[input             (masked)]: ")
#     f.write(lx666)
#     f.write('\n')
#     f.write("[output_small  (pridiction)]: ")
#     f.write(lys666)
#     f.write('\n')
#     f.write("[output_big    (pridiction)]: ")
#     f.write(ly666)
#     f.write('\n')
#     f.write("[label       (ground truth)]: ")
#     f.write(lg666)
#     f.write('\n')
#     f.write('\n')
#     # print("case",len(num),"!!!---------------------------------")
#     # print("[input             (masked)]:", lx666)
#     # print("[output_small  (pridiction)]:", lys666)
#     # print("[output_big    (pridiction)]:", ly666)
#     # print("[label       (ground truth)]:", lg666)
# #################################################################################################################################
