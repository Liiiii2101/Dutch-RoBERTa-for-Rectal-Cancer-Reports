#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Importing stock ml libraries
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig, RobertaTokenizer, RobertaModel, RobertaForTokenClassification, RobertaTokenizerFast
from sklearn.metrics import multilabel_confusion_matrix as mcm, classification_report  
from tqdm.auto import tqdm
from transformers import BertTokenizer, BertModel, BertConfig, RobertaTokenizer, RobertaTokenizerFast, RobertaForMaskedLM

###radiology
#path_train = ['/data/groups/beets-tan/l.cai/rectal_nlp/Train/Train_Radiology.xlsx', '/data/groups/beets-tan/l.cai/rectal_nlp/Test/Test_Radiology.xlsx']
global modality
global embedding

name_dict = {
  "RobBERT": "pdelobelle/robbert-v2-dutch-base",
  "MedRoBERTa.nl": "CLTL/MedRoBERTa.nl",
  "RecRoBERT": "/data/groups/beets-tan/l.cai/rectal_nlp/re_trained_model_only_rectal",
  "BRecRoBERT": "/data/groups/beets-tan/l.cai/rectal_nlp/Lishan_nlp",
  "BRec2RoBERT": "/data/groups/beets-tan/l.cai/rectal_nlp/trained_models",
  "BRoBERT": '/data/groups/beets-tan/l.cai/rectal_nlp/breast_only'
}

embedding = 'BRec2RoBERT'#'RobBERT'#'BRec2RoBERT'#'BRoBERT'#"BRecRoBERT"#"RecRoBERT"#"MedRoBERTa.nl"#'RobBERT'

model_path = name_dict[embedding]

modality = 'Radiology'#Pathology'#'Radiology'
#model_path = embedding#'/data/groups/beets-tan/l.cai/rectal_nlp/re_trained_model_only_rectal'#'/data/groups/beets-tan/l.cai/rectal_nlp/Lishan_nlp'#'pdelobelle/robbert-v2-dutch-base'#'CLTL/MedRoBERTa.nl'#'/data/groups/beets-tan/l.cai/rectal_nlp/re_trained_model_only_rectal'#'/data/groups/beets-tan/l.cai/rectal_nlp/trained_models'#'pdelobelle/robbert-v2-dutch-base'#'CLTL/MedRoBERTa.nl'#'/data/groups/beets-tan/l.cai/rectal_nlp/Lishan_nlp'#'./Lishan_nlp'#'CLTL/MedRoBERTa.nl'#'pdelobelle/robbert-v2-dutch-base'#'./re_trained_model_only_rectal'#'CLTL/MedRoBERTa.nl'#'pdelobelle/robbert-v2-dutch-base'#'./Lishan_nlp'
data_frame_path = '/data/groups/beets-tan/l.cai/rectal_nlp/df_tasks_reports_addmiss.csv'

df = pd.read_csv(data_frame_path, sep=',')

df_train = df[df['split'] == 'train']#pd.read_excel(path_train[0])
df_test = df[df['split'] == 'test']#pd.read_excel(path_train[1])

df_train = df_train[df_train[modality] != '0']
df_test = df_test[df_test[modality] != '0']


MAX_LEN = 512
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
LEARNING_RATE = 1e-05
#'pdelobelle/robbert-v2-dutch-base'
#/data/groups/beets-tan/l.cai/rectal_nlp/trained_models
#/data/groups/beets-tan/l.cai/rectal_nlp/Lishan_nlp
#/data/groups/beets-tan/l.cai/rectal_nlp/re_trained_model_only_rectal
tokenizer = RobertaTokenizerFast.from_pretrained(model_path, return_tensors='pt')#('/home/t.zhang/NLP/radiobert_BigDataset_epoch10', return_tensors='pt')


class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.title = dataframe['VERSLAG']

        self.max_len = max_len
        lb = [i.split() for i in dataframe['label'].values.tolist()]
        txt = dataframe['VERSLAG'].values.tolist()
        self.labels = [align_label(i,j) for i,j in zip(txt, lb)]

    def __len__(self):
        return len(self.title)
        
    def get_batch_labels(self, index):
        return torch.LongTensor(self.labels[index])

    def __getitem__(self, index):
        title = str(self.title[index])
        title = " ".join(title.split())

        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        
        #torch.tensor(self.labels[index], dtype=torch.long),

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'labels'  : self.get_batch_labels(index),
        }
        
class DataSequence(torch.utils.data.Dataset):

    def __init__(self, df):

        txt = df[modality].values.tolist()
        self.texts = [tokenizer(str(i),
                               padding='max_length', max_length = 512, truncation=True, return_tensors="pt") for i in txt]
        self.targets = df['id'].values.tolist()

    def __len__(self):

        return len(self.texts)#(self.targets)

    def get_batch_data(self, idx):

        return self.texts[idx]


    def __getitem__(self, idx):

        batch_data = self.get_batch_data(idx)


        return {
            'batch_data': batch_data,
            'batch_targets': self.targets[idx]#torch.tensor([0])#(self.targets[idx], dtype=torch.long),
        }

train_dataset = df_train.sample(frac=1,random_state=1203)
valid_dataset = df_test.sample(frac=1,random_state=1203)


print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(valid_dataset.shape))

train_dataset = DataSequence(df_train)
val_dataset = DataSequence(df_test)

#train_dataloader = DataLoader(train_dataset, num_workers=4, batch_size=1, shuffle=False)
dataloader_extractor_val = DataLoader(val_dataset, num_workers=4, batch_size=1)
dataloader_extractor_train = DataLoader(train_dataset, num_workers=4, batch_size=1)
print("train_dataloader:",len(dataloader_extractor_train))
print("val_dataloader:",len(dataloader_extractor_val))

import torch

# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
    
class Radio_RoBERTa(torch.nn.Module):

    def __init__(self):

        super(Radio_RoBERTa, self).__init__()

        #self.bert = RobertaModel.from_pretrained('/data/groups/beets-tan/l.cai/rectal_nlp/trained_models', add_pooling_layer=False) #/home/t.zhang/NLP/radiobert_BigDataset_epoch10
        
        #self.ldo = torch.nn.Dropout(0.3)
        #self.lbirads = torch.nn.Linear(768, 5)
        #self.check_load = torch.load("/data/groups/beets-tan/l.cai/rectal_nlp/trained_models/model.safetensors")#("/home/t.zhang/NLP/bert_features/best_model/best_model_big0712_accf1_weigthed.pt")
        self.bert = RobertaModel.from_pretrained(model_path, add_pooling_layer=False)#('/data/groups/beets-tan/l.cai/rectal_nlp/trained_models', add_pooling_layer=False)

    def forward(self, input_id=None, attention_mask=None): #, return_dict=None
        #self.bert.load_state_dict(self.check_load['state_dict'],strict=False)
        self.bert.eval()
        outputs = self.bert(input_ids=input_id, attention_mask=attention_mask, return_dict=False)
        sequence_output = outputs[0]
        #print('output shape', sequence_output.shape, len(outputs))
        pooler = sequence_output[:, 0]
        #do = self.ldo(pooler)
        #output666 = self.lbirads(do)

        return pooler#output666




def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()
    
import shutil, sys
def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)


#check_load = torch.load("/home/t.zhang/NLP/bert_features/best_model/best_model_big0713_accf1_weigthed.pt")
#model_birads=model.load_state_dict(check_load['state_dict'],strict=False)
# model.to(device)
# print(model)

checkpoint_path = '/home/t.zhang/NLP/bert_features/checkpoint/current_checkpoint_big0727_birads_tl.pt'
best_model = '/home/t.zhang/NLP/bert_features/best_model/best_model_big0727_birads_tl.pt'
#trained_model = train_model(1, 50, np.Inf, training_loader, validation_loader, model, optimizer,checkpoint_path,best_model)
#trained_model = train_loop(model, train_dataloader, val_dataloader,checkpoint_path, best_model)

print('---------------------------------------------------------------------------------------------------')

print("test: --------------------------------------------------------------------------------------------------------------------------------------------------------------------")

val_targets=[]
val_outputs=[]


model = Radio_RoBERTa()
#model = RobertaForMaskedLM.from_pretrained('/data/groups/beets-tan/l.cai/rectal_nlp/trained_models', return_dict=False)
model.to(device)

model.eval()

with torch.no_grad():

      loop_v=tqdm(dataloader_extractor_val,leave=True)
      for batch in loop_v:
            #print(batch.keys())
            mask = batch['batch_data']['attention_mask'][:,0,:].to(device)
            input_ids = batch['batch_data']['input_ids'][:,0,:].to(device)
            patient_id = batch['batch_targets'].item()
          
            
            #print('input id',input_id, mask)
            #targets = batch['batch_targets'].to(device, dtype = torch.long)              
            output = model(input_ids, mask).cpu().detach().numpy()
            #print(output.shape)
            #print(output)
            np.save(f"/data/groups/beets-tan/l.cai/rectal_nlp/{modality.lower()}_npy/test/{str(patient_id)}.npy", output)
            #print('output', output.shape, output)

            #val_targets.extend(targets.cpu().detach().numpy().tolist())
            #_, predicted = torch.max(output.data, dim=1)
            #val_outputs.extend(predicted.cpu().detach().numpy().tolist())

with torch.no_grad():

      loop_v=tqdm(dataloader_extractor_train,leave=True)
      for batch in loop_v:
            #print(batch.keys())
            mask = batch['batch_data']['attention_mask'][:,0,:].to(device)
            input_ids = batch['batch_data']['input_ids'][:,0,:].to(device)
            patient_id = batch['batch_targets'].item()
          
            
            #print('input id',input_id, mask)
            #targets = batch['batch_targets'].to(device, dtype = torch.long)              
            output = model(input_ids, mask).cpu().detach().numpy()
            #print(output.shape)
            #print(output)
            np.save(f"/data/groups/beets-tan/l.cai/rectal_nlp/{modality.lower()}_npy/train/{str(patient_id)}.npy", output)
            #print('output', output.shape, output)

            #val_targets.extend(targets.cpu().detach().numpy().tolist())
            #_, predicted = torch.max(output.data, dim=1)
            #val_outputs.extend(predicted.cpu().detach().numpy().tolist())


# print("classification_report_val_targets: BIRADS")
# labels_birads =['BIRADS 1', 'BIRADS 2', 'BIRADS 3', 'BIRADS 4', 'BIRADS 5']
# print(classification_report(val_targets, val_outputs, target_names=labels_birads))
# accuracy = metrics.accuracy_score(val_targets, val_outputs)
# f1_score_micro = metrics.f1_score(val_targets, val_outputs, average='micro')
# f1_score_macro = metrics.f1_score(val_targets, val_outputs, average='macro')
# f1_score_weighted = metrics.f1_score(val_targets, val_outputs, average='weighted')
# print(f"Accuracy Score = {accuracy}")
# print(f"F1 Score (Micro) = {f1_score_micro}")
# print(f"F1 Score (Macro) = {f1_score_macro}")
# print(f"F1 Score (Weighted) = {f1_score_weighted}")
# print("----------------------------------------------------------------------------------")
# matrix=metrics.confusion_matrix(val_targets, val_outputs)
# print(matrix) 

# from sklearn import metrics
# import pandas as pd
# import numpy as np

# import matplotlib as mpl
# import matplotlib.font_manager as fm
# import matplotlib.pyplot as plt


# def plot_matrix(y_test, y_pred, labels, title=None, thresh=0.5, axis_labels=None):
# # 利用sklearn中的函数生成混淆矩阵并归一化
#     cm0 = metrics.confusion_matrix(y_test, y_pred, sample_weight=None)  # 生成混淆矩阵 
#     cm = cm0.astype('float') / cm0.sum(axis=1)[:, np.newaxis]  # 归一化

# # 画图，如果希望改变颜色风格，可以改变此部分的cmap=pl.get_cmap('Blues')处
#     #plt.figure(figsize=(6,6))
#     plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
#     plt.colorbar()  # 绘制图例

# # 图像标题
#     if title is not None:
#         plt.title(title, fontsize=13)
# # 绘制坐标
#     num_local = np.array(range(len(labels)))
#     if axis_labels is None:
#         axis_labels = labels_name
#     plt.ylabel('Ground truth',size = 12)
#     plt.xlabel('Predict',size = 12)
#     plt.xticks(num_local, axis_labels, rotation=0, size = 8)  # 将标签印在x轴坐标上， 并倾斜45度
#     plt.yticks(num_local, axis_labels,size = 8)  # 将标签印在y轴坐标上,size = 20


# # 将百分比打印在相应的格子内，大于thresh的用白字，小于的用黑字
    
#     for i in range(np.shape(cm)[0]):
#         for j in range(np.shape(cm)[1]):
#             if int(cm[i][j]*100000 ) >= 0:
#                 plt.text(j, i, format(float(cm[i][j])*100,'.1f')+'%',  #+"\n"+"("+format(float(cm0[i][j]),'.0f')+")"
#                         ha="center", va="center",
#                         color="white" if cm[i][j] > thresh else "black")  # 如果要更改颜色风格，需要同时更改此行
# # 显示
#     plt.savefig('/home/t.zhang/NLP/bert_features/BI-RADS_new.png', dpi=300)
#     plt.show()
# y_test = val_targets
# y_pred = val_outputs
# plot_matrix(y_test, y_pred, [0,1,2,3,4], title=None, axis_labels=['BI-RADS 1','BI-RADS 2','BI-RADS 3','BI-RADS 4', 'BI-RADS 5'])
