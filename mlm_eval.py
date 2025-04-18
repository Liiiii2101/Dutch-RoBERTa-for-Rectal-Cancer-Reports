import os
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaTokenizer
import torch
from tqdm.auto import tqdm
from sklearn import metrics
from transformers import BertTokenizer, BertModel, BertConfig, RobertaTokenizer, RobertaTokenizerFast, RobertaForMaskedLM
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np
import random
import warnings
from sklearn.utils import resample

def bootstrap_accuracy_ci(y_true, y_pred, n_iterations=1000, ci_level=95):
    """
    Calculate the confidence interval for accuracy using bootstrapping.

    Parameters:
    - y_true: array-like, shape (n_samples,) True labels.
    - y_pred: array-like, shape (n_samples,) Predicted labels.
    - n_iterations: int, Number of bootstrap iterations (default=1000).
    - ci_level: int, Confidence level for the interval (default=95).

    Returns:
    - ci_low: float, Lower bound of the confidence interval.
    - ci_up: float, Upper bound of the confidence interval.
    - mean_accuracy: float, Mean accuracy from bootstrapped samples.
    """
    n_size = len(y_true)
    scores = []

    # Perform bootstrapping
    for _ in range(n_iterations):
        # Resample with replacement
        y_true_resampled, y_pred_resampled = resample(y_true, y_pred, n_samples=n_size)
        
        # Calculate accuracy on the resampled dataset
        score = metrics.accuracy_score(y_true_resampled, y_pred_resampled)
        scores.append(score)

    # Calculate mean accuracy
    mean_accuracy = np.mean(scores)

    # Calculate the confidence interval
    lower_percentile = (100 - ci_level) / 2
    upper_percentile = 100 - lower_percentile
    ci_low = np.percentile(scores, lower_percentile)
    ci_up = np.percentile(scores, upper_percentile)

    return ci_low, ci_up, mean_accuracy

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
warnings.filterwarnings("ignore")

class Dataset(torch.utils.data.Dataset):
  def __init__(self, encodings):
    self.encodings = encodings

  def __len__(self):
    return self.encodings['input_ids'].shape[0]

  def __getitem__(self, i):
    return {key: tensor[i] for key, tensor in self.encodings.items()}




paths = [str(x) for x in Path('./Test').glob('**/*.txt')]
#paths = ['./Test/endoscopy.txt']
tokenizer_small = RobertaTokenizer.from_pretrained('./Lishan_nlp', max_len=512)
tokenizer_big = RobertaTokenizer.from_pretrained('./trained_models', max_len=512)
tokenizer_dutch = RobertaTokenizer.from_pretrained('pdelobelle/robbert-v2-dutch-base')#AutoTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base")
tokenizer_raw = RobertaTokenizer.from_pretrained("CLTL/MedRoBERTa.nl")
tokenizer_scratch = RobertaTokenizer.from_pretrained('./re_trained_model_only_rectal')
tokenizer_5 = RobertaTokenizer.from_pretrained('./trained_models_5')
tokenizer_5_only_radiology = RobertaTokenizer.from_pretrained('./trained_models_5_only_radiology')
tokenizer_breast = RobertaTokenizer.from_pretrained('/data/groups/beets-tan/l.cai/rectal_nlp/breast_only')
#tokenizer_tf_general = RobertaTokenizer.from_pretrained('./trained_models_general_dutch')
#tokenizer_tf_medicine = RobertaTokenizer.from_pretrained('./trained_models_general_medicine')
#model = AutoModelForMaskedLM.from_pretrained("CLTL/MedRoBERTa.nl")
#model_raw = AutoModelForMaskedLM.from_pretrained("pdelobelle/robbert-v2-dutch-base")

def mlm(tensor, mask_radio=0.05):#mask_radio=0.15):
  rand = torch.rand(tensor.shape)
  mask_arr = (rand < mask_radio) * (tensor > 2)
  for i in range(tensor.shape[0]):
    selection = torch.flatten(mask_arr[i].nonzero()).tolist()
    tensor[i, selection] = 4
  return tensor

def mask_random(tokenizer_reader, paths, standard=None):

  lines = []

  for path in tqdm(paths[:]):
    with open(path, 'r', encoding='utf-8') as f:
      lines += f.read().split('\n')
  
  sample = tokenizer_reader(lines, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
  labels_ids = sample.input_ids
  attention_mask = sample.attention_mask
  if standard == None:
    input_ids = mlm(sample.input_ids.detach().clone())
  else:
    input_ids = torch.where(standard == 4, standard, labels_ids)

  return {
    'input_ids': input_ids[:,:],
    'attention_mask': attention_mask[:,:],
    'labels': labels_ids[:,:]} 




def eval_mlm(model, data, device='cuda',model_name='general dutch'):

  XX_small=[]
  YY_small=[]
  TOP5_small=[]
  TOP10_small=[]
  output_viz_small=[]
  input_viz_small=[]
  labels_viz_small=[]

  model.to(device)
  model.eval()

  with torch.no_grad():
    loop = tqdm(data, leave=True)
  
    for batch in loop:
      input_ids6 = batch['input_ids'].to(device)
      mask6 = batch['attention_mask'].to(device)
      labels6 = batch['labels'].to(device)

      outputs = model(input_ids6, attention_mask=mask6)
      outputs123=torch.flatten(outputs[0],start_dim=-1)  #!!!
      b123=torch.reshape(outputs123, (outputs123.shape[0]*outputs123.shape[1],outputs123.shape[2]))

      y=torch.argmax(outputs[0], -1)
      x=input_ids6
      z=labels6

      y123=torch.flatten(y)  #!!!
      x123=torch.flatten(x)  #!!!
      z123=torch.flatten(z)  #!!!

      b5=torch.topk(b123, 5)[1]
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
      T10=[]
      for i in range(512*outputs123.shape[0]):
        if z123[i] in b10[i]:
          r1=z123[i].tolist()
          T10.append(r1)
        else:
          r2=b10[i][0].tolist()
          T10.append(r2)

      t10=torch.LongTensor(T10).to(device)
      p=torch.where(x123 == 4, y123, 0)
      g=torch.where(x123 == 4, z123, 0)
      t_5=torch.where(x123 == 4, t5, 0)
      t_10=torch.where(x123 == 4, t10, 0)

      pp= p.tolist()
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

        ly=out_small[i]
        ly1=ly.tolist()

        lg=labels6[i]
        lg1=lg.tolist()
        LX.append(lx1)
        LY.append(ly1)
        LG.append(lg1)

      input_viz_small.append(LX)
      output_viz_small.append(LY)
      labels_viz_small.append(LG)

    outputs_small=sum(XX_small, [])
    gt_small=sum(YY_small, [])
    top5_small=sum(TOP5_small, [])
    top10_small=sum(TOP10_small, [])

    print("output_small        len:", len(outputs_small))
    print("groundtruth_small   len:", len(gt_small))
    print("top5_small          len:", len(top5_small))
    print("top10_small         len:", len(top10_small))

    if len(outputs_small) == len(gt_small) == len(top5_small) == len(top10_small):
      print("Num is equal!       len:", len(outputs_small))

    else:
      print("Error!!!")

    low1, up1, accuracy_small_top1 = bootstrap_accuracy_ci(gt_small, outputs_small)#metrics.accuracy_score(outputs_small, gt_small)
    low5, up5, accuracy_small_top5 = bootstrap_accuracy_ci(gt_small, top5_small)#metrics.accuracy_score(top5_small, gt_small)
    low10, up10, accuracy_small_top10 = bootstrap_accuracy_ci(gt_small, top10_small)#metrics.accuracy_score(top10_small, gt_small)
    print(f"model           name: {model_name}")
    print("total            num:", len(outputs_small))
    print("correct          num:", int(accuracy_small_top1*len(outputs_small)))
    print("------------------------")
    print("accuracy_small_top1 :", "%.5f" % accuracy_small_top1, low1, up1)
    print("accuracy_small_top5 :", "%.5f" % accuracy_small_top5, low5, up5)
    print("accuracy_small_top10:", "%.5f" % accuracy_small_top10, low10, up10)

    input_viz_small_c=sum(input_viz_small, [])
    output_viz_small_c=sum(output_viz_small, [])
    labels_viz_small_c=sum(labels_viz_small, [])




encodings_small = mask_random(tokenizer_small, paths)
dataset_small = Dataset(encodings_small)
dataloader_small = torch.utils.data.DataLoader(dataset_small, batch_size=64, shuffle=False)

encodings_big = mask_random(tokenizer_big, paths, standard=encodings_small['input_ids'])
dataset_big = Dataset(encodings_big)
dataloader_big = torch.utils.data.DataLoader(dataset_big, batch_size=64, shuffle=False)

encodings_raw = mask_random(tokenizer_raw, paths, standard=encodings_small['input_ids'])
dataset_raw = Dataset(encodings_raw)
dataloader_raw = torch.utils.data.DataLoader(dataset_raw, batch_size=64, shuffle=False)

encodings_dutch = mask_random(tokenizer_dutch, paths, standard=encodings_small['input_ids'])
dataset_dutch = Dataset(encodings_dutch)
dataloader_dutch = torch.utils.data.DataLoader(dataset_dutch, batch_size=64, shuffle=False)

encodings_scratch = mask_random(tokenizer_scratch, paths, standard=encodings_small['input_ids'])
dataset_scratch = Dataset(encodings_scratch)
dataloader_scratch = torch.utils.data.DataLoader(dataset_scratch, batch_size=64, shuffle=False)

encodings_5 = mask_random(tokenizer_5, paths, standard=encodings_small['input_ids'])
dataset_5 = Dataset(encodings_5)
dataloader_5 = torch.utils.data.DataLoader(dataset_5, batch_size=64, shuffle=False)

encodings_5_only_radiology = mask_random(tokenizer_5_only_radiology, paths, standard=encodings_small['input_ids'])
dataset_5_only_radiology = Dataset(encodings_5_only_radiology)
dataloader_5_only_radiology = torch.utils.data.DataLoader(dataset_5_only_radiology, batch_size=64, shuffle=False)


encodings_breast = mask_random(tokenizer_breast, paths, standard=encodings_small['input_ids'])
dataset_breast = Dataset(encodings_breast)
dataloader_breast = torch.utils.data.DataLoader(dataset_breast, batch_size=64, shuffle=False)
# encodings_tf_general = mask_random(tokenizer_tf_general, paths, standard=encodings_small['input_ids'])
# dataset_tf_general = Dataset(encodings_tf_general)
# dataloader_tf_general = torch.utils.data.DataLoader(dataset_tf_general, batch_size=64, shuffle=False)

# encodings_tf_medicine = mask_random(tokenizer_tf_medicine, paths, standard=encodings_small['input_ids'])
# dataset_tf_medicine = Dataset(encodings_tf_medicine)
# dataloader_tf_medicine = torch.utils.data.DataLoader(dataset_tf_medicine, batch_size=64, shuffle=False)
#################################################################################################################################
### radiobert general dutch

model_raw = RobertaForMaskedLM.from_pretrained('CLTL/MedRoBERTa.nl')#("pdelobelle/robbert-v2-dutch-base")
model_small=RobertaForMaskedLM.from_pretrained('./Lishan_nlp', return_dict=False)
model_big= RobertaForMaskedLM.from_pretrained('./trained_models', return_dict=False)
model_dutch = RobertaForMaskedLM.from_pretrained("pdelobelle/robbert-v2-dutch-base")
model_scratch = RobertaForMaskedLM.from_pretrained('./re_trained_model_only_rectal')
model_5 = RobertaForMaskedLM.from_pretrained('./trained_models_5', return_dict=False)
model_5_only_radiology = RobertaForMaskedLM.from_pretrained('./trained_models_5_only_radiology', return_dict=False)
model_breast = RobertaForMaskedLM.from_pretrained('/data/groups/beets-tan/l.cai/rectal_nlp/breast_only', return_dict=False)
#model_tf_general = RobertaForMaskedLM.from_pretrained('./trained_models_general_dutch')
#model_tf_medicine = RobertaForMaskedLM.from_pretrained('./trained_models_general_medicine')

#eval_mlm(model_tf_medicine, dataloader_tf_medicine, model_name='Transfer Learning from general medicine model')
#eval_mlm(model_tf_general, dataloader_tf_general, model_name='Transfer Learning from general dutch model')
eval_mlm(model_breast, dataloader_breast, model_name='breast only')
del model_breast
eval_mlm(model_scratch, dataloader_scratch, model_name='Train from scratch')
del model_scratch
eval_mlm(model_dutch, dataloader_dutch, model_name='General Dutch')
del model_dutch
del dataloader_dutch
eval_mlm(model_raw, dataloader_raw, model_name='General Dutch Medicine')
del model_raw
del dataloader_raw
eval_mlm(model_big, dataloader_big, model_name='Rectal')
del model_big
del dataloader_big
eval_mlm(model_small, dataloader_small, model_name='Breast')
del model_small
eval_mlm(model_5, dataloader_5, model_name='retrain 5 epochs all rectal data')
del model_5
eval_mlm(model_5_only_radiology, dataloader_5_only_radiology, model_name='retrain 5 epochs all rectal data')


