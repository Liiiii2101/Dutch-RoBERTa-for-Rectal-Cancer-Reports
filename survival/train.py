import os
import numpy as np
import pandas as pd
import torch
import logging
import argparse
from torch import optim
from pathlib import Path
from sklearn.model_selection import KFold
#from fastai.data.load import DataLoader
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from mil.model import MILModel
from Early_Stopping import EarlyStopping
from loss import cox_loss, concordance_index
from sklearn.model_selection import train_test_split
from mil.data import get_cohort_df, make_dataset, DataSequence
from sklearn.model_selection import train_test_split
import warnings
import random
from collections import Counter

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Train')
#parser.add_argument('-ct', '--clinical_table', default='/data/groups/beets-tan/l.cai/rectal_nlp/Pateint_OS.csv',type=Path, required=True, help='clinical_table')
#parser.add_argument('-st', '--slide_table', default='/data/groups/beets-tan/l.cai/rectal_nlp/radiology_npy',type=Path, required=True, help='slide_table')
parser.add_argument('-f', '--feature_dir', default='/data/groups/beets-tan/l.cai/rectal_nlp/radiology_npy',type=Path,  help='feature_dir')
parser.add_argument('-o', '--output_path', default='/data/groups/beets-tan/l.cai/rectal_nlp/output',type=Path, help='output_path')
parser.add_argument('-t', '--target_label', default=['os', 'os_e'], nargs='+', type=str,  help='target_label, e.g., [os, os_e]')
parser.add_argument('-bs', '--batch_size', default=4, type=int, help='batch_size')
parser.add_argument('-nw', '--num_workers', default=8, type=int, help='num_workers')
parser.add_argument('-ep', '--epochs', default=100, type=int, help='epochs')
parser.add_argument('-lr', '--lr', default=1e-5, type=float, help='lr')
parser.add_argument('-bgs', '--bag_size', default=2, type=int, help='bag_size')
parser.add_argument('-l1', '--l1_reg', default=1e-3, type=float, help='l1_reg')
parser.add_argument('-l2', '--l2_reg', default=1e-3, type=float, help='l2_reg')
parser.add_argument('-embed', default='RobBERT', type=str)
parser.add_argument('-goal', default="OS", type=str)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def get_table(clini_excel, slide_csv, feature_dir, target_label, output_path):

    test_patients = None
    df = get_cohort_df(clini_excel, slide_csv, feature_dir, target_label, categories=None)

    if os.path.exists(output_path / 'train.csv') and os.path.exists(output_path / 'test.csv'):
        train_patients = pd.read_csv(output_path / 'train.csv').PATIENT
        test_patients = pd.read_csv(output_path / 'test.csv').PATIENT
        valid_patients = pd.read_csv(output_path / 'valid.csv').PATIENT
    elif os.path.exists(output_path / 'train.csv') and not os.path.exists(output_path / 'test.csv'):
        train_patients = pd.read_csv(output_path / 'train.csv').PATIENT
        valid_patients = pd.read_csv(output_path / 'valid.csv').PATIENT
    else:
        train_patients, valid_patients = train_test_split(df.PATIENT, test_size=0.2)

        train_df = df[df.PATIENT.isin(train_patients)]
        valid_df = df[df.PATIENT.isin(valid_patients)]

        print(f'train:{len(train_df)}')
        print(f'valid:{len(valid_df)}')

        train_df.drop(columns='slide_path').to_csv(output_path / 'train.csv', index=False)
        valid_df.drop(columns='slide_path').to_csv(output_path / 'valid.csv', index=False)
    return df, train_patients, valid_patients, test_patients


def cal_loss(y, pred, criterion):
    loss_dict = {}
    
    for target_class in range(int(y.shape[1] / 2)):
        loss_dict[target_class] = criterion(y[:, target_class * 2:(target_class + 1) * 2].to(device), pred.to(device))
    return loss_dict


def cal_ci(y, pred):
    ci_dict = {}
    for target_class in range(int(y.shape[1] / 2)):
        ci_dict[target_class] = concordance_index(y[:, target_class * 2:(target_class + 1) * 2].to(device),
                                                  -pred.to(device))
    return ci_dict


def prediction(model, dl, criterion):
    with torch.no_grad():
        model.eval()
        #for i, data in enumerate(dl):
        for i, batch_data in enumerate(dl):
            x_batch = batch_data[0]
            len_batch = batch_data[1]
            y_batch = batch_data[2]

            pred = model(x_batch.to(device), len_batch.to(device))
            # feature = data['features']
            # y_batch = data['labels']
            # x_batch_len = torch.full((feature.shape[0],), 1, dtype=torch.int32)#torch.tensor([1]*1)
            # idi = data['id']
            
            #print(feature.shape, x_batch_len.shape)
            #print(pred,y_batch)
            #x_batch = batch_data[0]
            #len_batch = batch_data[1]
            #y_batch = batch_data[2]

            #pred = model(feature.to(device), x_batch_len.to(device))
            #print(pred,y_batch)

            if i == 0:
                pred_all = pred
                y_all = y_batch

            else:
                pred_all = torch.cat([pred_all, pred])
                y_all = torch.cat([y_all, y_batch])

        loss_dict = cal_loss(y_all, pred_all, criterion)
        ci_dict = cal_ci(y_all, pred_all)

    return loss_dict, ci_dict, pred_all


def train(epochs, model, train_dl, device, criterion,  valid_dl, logger, l1, l2, test_dl):
    for epoch in range(epochs):
        logger.info("\nStart of epoch %d" % (epoch,))

        model.train()

        #for data in train_dl:
        for x_batch_train, x_batch_len, y_batch_train in train_dl:
            
            #print(x_batch_train[0].shape[-1], y_batch_train)
            #print(x_batch_len, x_batch_train.shape)
            # feature = data['features']
            # label = data['labels']
            # x_batch_len = torch.full((feature.shape[0]+1,), 1, dtype=torch.int32)
            # idi = data['id']
            # print(feature.shape, feature[0].shape)
            optimizer.zero_grad()


            outputs = model(x_batch_train.to(device), x_batch_len.to(device))
            loss = sum(cal_loss(y_batch_train, outputs, criterion).values())

            l1_reg = torch.tensor(0.).to(device)
            for param in model.parameters():
                l1_reg += torch.abs(param).sum()

            l2_reg = torch.tensor(0.).to(device)
            for param in model.parameters():
                l2_reg += torch.norm(param)

            loss = loss + l1 * l1_reg + l2 * l2_reg
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()

            train_loss_dict, train_ci_dict, train_score = prediction(model, train_dl, criterion)
            valid_loss_dict, valid_ci_dict, valid_score = prediction(model, valid_dl, criterion)
            
            test_loss_dict, test_ci_dict, test_score = prediction(model, test_dl, criterion)



        # ci_total = valid_os_ci + valid_dfs_ci
        ci_total = sum(valid_ci_dict.values())
        scheduler.step()
        #scheduler.step(next(iter(valid_loss_dict.values())))
        #scheduler.step(valid_loss_dict.values()[0])
        early_stopping(-ci_total, model)

        if early_stopping.early_stop:
            logger.info("Early stopping")
            break

        for key in train_ci_dict.keys():
            logger.info(f'train ci_{key}: {train_ci_dict[key]}')
        for key in valid_ci_dict.keys():
            logger.info(f'valid ci_{key}: {valid_ci_dict[key]}')

        
        logger.info(f'median score {np.median(train_score.cpu().detach())}')
        logger.info(f'median score val {np.median(valid_score.cpu().detach())}')
        logger.info('--------------------------------')


if __name__ == '__main__':
    args = parser.parse_args()
    #clini_excel = args.clinical_table
    #slide_csv = args.slide_table
    feature_dir = args.feature_dir
    output_path = args.output_path
    target_label = args.target_label

    assert len(target_label) % 2 == 0, "The length of target label should be even"
    global modality
    global goal
    modality = 'Radiology'
    goal = args.goal#'OS' or 'DFS'
    embedding = args.embed
    batch_size = args.batch_size
    num_workers = args.num_workers
    epochs = args.epochs
    lr = args.lr
    bag_size = 2#args.bag_size
    l1 = args.l1_reg
    l2 = args.l2_reg

    model_save_path = output_path / f'lr_{lr}_l1_{l1}_l2_{l2}_best_model_miss_{modality}_{goal}.pth'

    feature_dir = Path(feature_dir)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DACHS = True
    #df, train_patients, valid_patients, test_patients = get_table(clini_excel, slide_csv, feature_dir, target_label,
    #                                                              output_path)

    # targs = df[target_label].values
    # bags = df.slide_path.values
    # valid_idxs = df.PATIENT.isin(valid_patients).values
    # train_idx = df.PATIENT.isin(train_patients).values

    logger = get_logger(output_path / f'lr_{lr}_l1_{l1}_l2_{l2}_exp_log_miss_{modality}_{goal}')
    logger.info('Overall distribution')
    logger.info(f'batch_size:{batch_size}')
    logger.info(f'lr:{lr}')
    logger.info(f'DACHS:{DACHS}')
    logger.info(f'l1 lambda:{l1}')
    logger.info(f'l2 lambda:{l2}')

    add_features = []

    data_frame_path = '/data/groups/beets-tan/l.cai/rectal_nlp/df_tasks_reports_addmiss.csv'

    df = pd.read_csv(data_frame_path, sep=',')
    df = df[df[f'{goal} Time'].notna()]
    df = df[df[f'{goal}_event'] >= 0]
    df_test = df[df['split'] == 'test']
    df= df[df['split'] == 'train']
   

    df = df[df[modality] != '0' ]
    df_test = df_test[df_test[modality] != '0']
  

    modality = modality.lower()
    
    adding_pathology = False

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 0

    for fold, (train_ids, test_ids) in enumerate(kf.split(list(df['id']))):
        train_ids = np.array(df['id'])[train_ids]
        test_ids = np.array(df['id'])[test_ids]
        train_dataset = df[df['id'].isin(train_ids)]
        val_dataset  = df[df['id'].isin(test_ids)]

        print(f"Training of Fold {fold}")
        add_features = []
        train_bags = [[f"/data/groups/beets-tan/l.cai/rectal_nlp/{modality}_npy/train/{i}.npy"] for i in list(train_dataset['id'])]

        train_targs = [[train_dataset[train_dataset['id']==i][f'{goal} Time'].values, train_dataset[train_dataset['id']==i][f'{goal}_event'].values] for i in list(train_dataset['id'])]

        val_bags = [[f"/data/groups/beets-tan/l.cai/rectal_nlp/{modality}_npy/train/{i}.npy"] for i in list(val_dataset['id'])]

        val_targs = [[val_dataset[val_dataset['id']==i][f'{goal} Time'].values, val_dataset[val_dataset['id']==i][f'{goal}_event'].values] for i in list(val_dataset['id'])]

       


        if adding_pathology:
            for j,i in enumerate(list(train_dataset['id'])):
                if train_dataset[train_dataset['id'] == i]['Pathology'].values != '0':
                    train_bags[j].append(f"/data/groups/beets-tan/l.cai/rectal_nlp/pathology_npy/train/{i}.npy")
            for j,i in enumerate(list(val_dataset['id'])):
                if val_dataset[val_dataset['id'] == i]['Pathology'].values != '0':
                    val_bags[j].append(f"/data/groups/beets-tan/l.cai/rectal_nlp/pathology_npy/train/{i}.npy")
            for j,i in enumerate(list(df_test['id'])):
                if df_test[df_test['id'] == i]['Pathology'].values != '0':
                    test_bags[j].append(f"/data/groups/beets-tan/l.cai/rectal_nlp/pathology_npy/test/{i}.npy")
            
       
        bag_size = 1
        train_ds = make_dataset(
            bags=train_bags, 
            targets=train_targs, 
            add_features=[
                (enc, vals[train_idx])
                for enc, vals in add_features],
            bag_size=bag_size)

        train_ds_v = make_dataset(
            bags=train_bags, 
            targets=train_targs,
            add_features=[
                (enc, vals[train_idx])
                for enc, vals in add_features],
            bag_size=bag_size)

        valid_ds = make_dataset(
            bags=val_bags, 
            targets=val_targs,
            add_features=[
                (enc, vals[valid_idxs])
                for enc, vals in add_features],
            bag_size=bag_size)

        test_ds = make_dataset(
            bags=test_bags, 
            targets=test_targs,
            add_features=[
                (enc, vals[valid_idxs])
                for enc, vals in add_features],
            bag_size=bag_size)


        
        model_save_path = output_path / f'lr_{lr}_l1_{l1}_l2_{l2}_best_model_miss_{modality}_{goal}_fold{fold}_{embedding}_5cv.pth'






    # build dataloaders
        drop_last = True  


        train_dl = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last,
            num_workers=num_workers)

        train_dl_v = DataLoader(
            train_ds_v, batch_size=1, shuffle=False,
            num_workers=os.cpu_count())

        valid_dl = DataLoader(
            valid_ds, batch_size=1, shuffle=False,
            num_workers=os.cpu_count())

        test_dl = DataLoader(
            test_ds, batch_size=1, shuffle=False,
            num_workers=os.cpu_count())



        model = MILModel(768, 1)

        optimizer = optim.Adam(model.parameters(), lr=lr)

        criterion = cox_loss

        model = model.to(device)

        early_stopping = EarlyStopping(model_path=model_save_path,
                                    patience=10, verbose=True)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1, last_epoch=-1, verbose=True)


        train(epochs, model, train_dl, device, criterion,  valid_dl, logger, l1, l2, test_dl)
