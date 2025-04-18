import os
import torch
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from loss import cox_loss
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from mil.model import MILModel
from mil.data import make_dataset, get_cohort_df
from train import prediction, get_logger
from mil.data import get_cohort_df, make_dataset, DataSequence
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import scienceplots
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
#plt.style.use(['science','no-latex'])
from sklearn.utils import resample
from sksurv.metrics import concordance_index_censored
import re



def bootstrap_cindex_ci(event_times, event_observed, risk_scores, n_iterations=1000, ci_level=95):
    """
    Calculate the confidence interval for the c-index using bootstrapping.

    Parameters:
    - event_times: array-like, shape (n_samples,) Time to event or censoring.
    - event_observed: array-like, shape (n_samples,) Boolean array, 1 if event occurred, 0 if censored.
    - risk_scores: array-like, shape (n_samples,) Predicted risk scores.
    - n_iterations: int, Number of bootstrap iterations (default=1000).
    - ci_level: int, Confidence level for the interval (default=95).

    Returns:
    - ci_low: float, Lower bound of the confidence interval.
    - ci_up: float, Upper bound of the confidence interval.
    - mean_cindex: float, Mean c-index from bootstrapped samples.
    """
    n_size = len(event_times)
    cindex_scores = []

    # Perform bootstrapping
    for _ in range(n_iterations):
        # Resample with replacement
        event_times_resampled, event_observed_resampled, risk_scores_resampled = resample(
            event_times, event_observed, risk_scores, n_samples=n_size)
        
        # Calculate the c-index on the resampled dataset
        cindex = concordance_index_censored(
            event_observed_resampled, event_times_resampled, risk_scores_resampled
        )[0]
        cindex_scores.append(cindex)

    # Calculate mean c-index
    mean_cindex = np.mean(cindex_scores)

    # Calculate the confidence interval
    lower_percentile = (100 - ci_level) / 2
    upper_percentile = 100 - lower_percentile
    ci_low = np.percentile(cindex_scores, lower_percentile)
    ci_up = np.percentile(cindex_scores, upper_percentile)

    return ci_low, ci_up, mean_cindex

global modality
global goal

modality = 'Radiology'
# goal = 'DFS'#'OS'
#embedding = "RobBERT"
parser = argparse.ArgumentParser(description='Train')
#parser.add_argument('-ct', '--clinical_table', type=Path, required=True, help='clinical_table')
#parser.add_argument('-st', '--slide_table', type=Path, required=True, help='slide_table')
#parser.add_argument('-f', '--feature_dir', type=Path, required=True, help='feature_dir')
parser.add_argument('-o', '--output_path', default='/data/groups/beets-tan/l.cai/rectal_nlp/output',type=Path, help='output_path')
parser.add_argument('-t', '--target_label', default=['os', 'os_e'], nargs='+', type=str,  help='target_label, e.g., [os, os_e]')
parser.add_argument('-m', '--model_path', default=f'/data/groups/beets-tan/l.cai/rectal_nlp/output/lr_1e-05_l1_0.001_l2_0.001_best_model_miss_radiology_DFS_fold0_trained_models.pth',type=Path, help='model_path')
parser.add_argument('-embed', default='RobBERT', type=str)
parser.add_argument('-median', default=0.27, type=float)
parser.add_argument('-goal', default="OS", type=str)
parser.add_argument('-fold',default=0, type=int)
#parser.add_argument('-t', '--target_label', nargs='+', type=str, required=True, help='target_label, e.g., [os, os_e]')
parser.add_argument('-c', '--cohort', type=str, default='dfs', help='cohort name')


if __name__ == '__main__':
    args = parser.parse_args()
    #clini_excel = args.clinical_table
    #slide_csv = args.slide_table
    #feature_dir = args.feature_dir
    output_path = args.output_path
    model_path = args.model_path
    target_label = args.target_label
    cohort = args.cohort
    embedding = args.embed#"BRec2RoBERT"#"RecRoBERT"#"MedRoBERTa.nl"#"RobBERT"#"BRecRoBERT"#"MedRoBERTa.nl"#"RobBERT"#'trained_models_2'
    goal = args.goal
    fold = args.fold
##one year

    # if not os.path.exists(output_path):
    #     os.mkdir(output_path)

    #feature_dir = Path(feature_dir)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #df = get_cohort_df(clini_excel, slide_csv, feature_dir, target_label, categories=None)

    logger = get_logger(output_path / f'{cohort}_eval_exp.log')
    #logger.info(f'test:{len(df)}')

    add_features = []

    data_frame_path = '/data/groups/beets-tan/l.cai/rectal_nlp/df_tasks_reports_addmiss.csv'
    
    df_test = df[df['split'] == 'test']#pd.read_excel(path_train[1])
    df_train = df[df['split'] == 'train']

   
    print('overall: ', Counter(list(df_test[f'{goal}_event'])))
    df_test = df_test[df_test[f'{modality}'] != '0']
    df_train = df_train[df_train[f'{modality}'] != '0']
    test_dataset = DataSequence(df_test, train=False)

    train_ids, test_ids = train_test_split(list(df_train['id']), test_size=0.2, random_state=42)
    train_dataset = df_train[df_train['id'].isin(train_ids)]
    val_dataset  = df_train[df_train['id'].isin(test_ids)]
    #df_test = train_dataset

    train_bags = [[f"/data/groups/beets-tan/l.cai/rectal_nlp/{modality}_npy/train/{i}.npy"] for i in list(train_dataset['id'])]
    
    train_targs = [[train_dataset[train_dataset['id']==i][f'{goal} Time'].values, train_dataset[train_dataset['id']==i][f'{goal}_event'].values] for i in list(train_dataset['id'])]

    val_bags = [[f"/data/groups/beets-tan/l.cai/rectal_nlp/{modality}_npy/train/{i}.npy"] for i in list(val_dataset['id'])]

    val_targs = [[val_dataset[val_dataset['id']==i][f'{goal} Time'].values, val_dataset[val_dataset['id']==i][f'{goal}_event'].values] for i in list(val_dataset['id'])]

    test_bags = [[f"/data/groups/beets-tan/l.cai/rectal_nlp/{modality.lower()}_npy/test/{i}.npy"] for i in list(df_test['id'])]

    test_targs = [[df_test[df_test['id']==i][f'{goal} Time'].values, df_test[df_test['id']==i][f'{goal}_event'].values] for i in list(df_test['id'])]

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

    drop_last = True  
    batch_size = 1
    num_workers=2

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
    model_path = f"/data/groups/beets-tan/l.cai/rectal_nlp/output/lr_1e-05_l1_0.001_l2_0.001_best_model_miss_radiology_{goal}_fold{str(fold)}_{embedding}_5cv.pth"
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    criterion = cox_loss
    with torch.no_grad():
        model.eval()

        test_loss_dict, test_ci_dict, test_score = prediction(model, test_dl, criterion)

    for key in test_ci_dict.keys():
        logger.info(f'test ci_{key}: {test_ci_dict[key]}')


    score = test_score.cpu().detach().numpy()
    logger.info(np.median(score))
    test_df = df_test.reset_index(drop=True)
    test_df['SCORE'] = list(score.flatten())
    median_score =args.median


    test_df.to_csv(output_path / f'{cohort}_score.csv')
    test_df['group'] = test_df['SCORE']
    test_df['group'] = test_df['group'].apply(lambda x: 'high' if x>=median_score else 'low')

    vis_df = test_df.copy()


    high = test_df[test_df['SCORE'] >=median_score][[f'{goal} Time', f'{goal}_event']]
    low = test_df[test_df['SCORE'] < median_score][[f'{goal} Time', f'{goal}_event']]
    test_df['bool_event'] = test_df[f'{goal}_event'].apply(lambda x: bool(x))

    ci_low, ci_up, mean_cindex = bootstrap_cindex_ci(test_df[f'{goal} Time'], test_df['bool_event'], test_df['SCORE'], n_iterations=1000, ci_level=95)

print('confidential interval:', ci_low, ci_up, 'mean C-Index:' ,mean_cindex)


print('high',high.shape)
print('low',low.shape)
from matplotlib import font_manager as fm
path = '/data/groups/beets-tan/l.cai/rectal_nlp/font/arial.ttf'  # Update this path to where Arial is located
prop = fm.FontProperties(fname=path, size=14)







##one year
import matplotlib.pyplot as plt666
test_df.to_csv(f'csvs/{goal}_{embedding}_fold{str(fold)}.csv')
plt666.figure(figsize=(8, 8))
test_dfa = test_df[(test_df['OS_event']==1) | (test_df['OS Time']>=36)]
print(test_dfa.shape)
ostime = np.array(list(test_dfa['OS Time']))
score = np.array(list(test_dfa['SCORE']))
ostime = [0 if i >= 12 else 1 for i in ostime]
fpr, tpr, _ = roc_curve(ostime, score)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt666.figure()
lw = 2
plt666.plot(fpr, tpr, color='darkblue',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt666.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
plt666.xlim([0.0, 1.0])
plt666.ylim([0.0, 1.05])
plt.xticks(fontsize=6)  
plt.yticks(fontsize=6)
plt666.xlabel('False Positive Rate',fontproperties=prop, fontsize=12)
plt666.ylabel('True Positive Rate',fontproperties=prop, fontsize=12)
plt666.title('3-year Survival', fontproperties=prop, fontsize=12)
plt666.legend(loc="lower right",  fontsize=12)
plt666.tight_layout()
#plt666.savefig(f'{goal}_{embedding}_auc_threeyear.png', dpi=600)


high = high.reset_index(drop=True)
low = low.reset_index(drop=True)
censor_styles = {'marker': '|', 'ms': 8, 'mew': 1}
# Calculate OS curve
kmf_os_high = KaplanMeierFitter()
kmf_os_high.fit(high[f'{goal} Time'], event_observed=high[f'{goal}_event'], label='High')

kmf_os_low = KaplanMeierFitter()
kmf_os_low.fit(low[f'{goal} Time'], event_observed=low[f'{goal}_event'], label='Low')

#censor_styles = {'marker': '|', 'mew': 2, 'markersize': 8}
plt666.figure(figsize=(8, 6))
kmf_os_high.plot(ci_show=False, color='red', linestyle='-', show_censors=True, censor_styles=censor_styles, linewidth=3.5)
kmf_os_low.plot(ci_show=False, color='blue', linestyle='-', show_censors=True, censor_styles=censor_styles, linewidth=3.5)
# plt666.title(f'Test set',fontproperties=prop, fontsize=16)
# plt666.xlabel('Time (months)', fontsize=16, fontproperties=prop)
# plt666.ylabel(f'OS', fontsize=16, fontproperties=prop)
# plt666.gca().spines['right'].set_visible(False)
# plt666.gca().spines['top'].set_visible(False)
# plt666.ylim((0.,1.0))
# plt666.xlim((0.,100))
# plt666.legend(prop=prop, fontsize=24)
# hr_text = 'Low risk: reference\nHR: 2.34 (95%CI 1.32-4.16), p=0.0036\nLog-rank test p=0.0027'
# plt666.text(2, 0.05, hr_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.5),fontproperties=prop)
# plt666.tight_layout()
# plt666.savefig(f'{goal} Kaplan-Meier Curve noCI add miss {modality}_{embedding}.png', dpi=600)

# high = high.rename(columns = {'OS Time': 'duration', 'OS_event': 'event'})
# low = low.rename(columns={'OS Time': 'duration', 'OS_event': 'event'})
result = logrank_test(high[f'{goal} Time'], low[f'{goal} Time'], event_observed_A=high[f'{goal}_event'],
                          event_observed_B=low[f'{goal}_event'])
p_value = result.p_value


print('log rank', p_value)



cph = CoxPHFitter()


test_df['group'] = test_df['SCORE'].apply(lambda x: 1 if x >= median_score else 0)

#test_df['group'] = test_df_t['T'].apply(lambda x: 1 if x>=3 else 0)
#test_df['group'] = test_df_t['N'].apply(lambda x: 1 if x>=1 else 0)
test_df = test_df.dropna(subset=['group'])
print(Counter(list(test_df['group'])), len(list(test_df['group'])))
cph_frame = test_df[[f'{goal} Time', f'{goal}_event', 'group']]
cph_frame = cph_frame.reset_index()
cph_frame['group'] = cph_frame['group'].astype('category').cat.codes
print(Counter(cph_frame[f'{goal}_event']), Counter(cph_frame['group']))


cph.fit(cph_frame, duration_col=f'{goal} Time', event_col=f'{goal}_event', formula="group")


cph.print_summary(decimals=6)
print("-----------------------")
a666 = cph.summary["p"].round(6).to_numpy().tolist()[0]
hr = cph.summary['exp(coef)'].round(6).to_numpy().tolist()[0]
hr95_low = cph.summary['exp(coef) lower 95%'].round(6).to_numpy().tolist()[0]
hr95_high = cph.summary['exp(coef) upper 95%'].round(6).to_numpy().tolist()[0]

print("-----------------------")

import matplotlib.pyplot as plt666

plt666.title(f'{embedding}',fontproperties=prop, fontsize=22)
#plt666.xlabel('Time (months)', fontsize=22, fontproperties=prop)
plt666.xlabel('', fontsize=22, fontproperties=prop)
plt666.ylabel(f'{goal}', fontsize=22, fontproperties=prop)
plt666.gca().spines['right'].set_visible(False)
plt666.gca().spines['top'].set_visible(False)
plt666.ylim((0.,1.0))
plt666.xlim((0.,100))

plt666.legend(prop=prop, fontsize=24)
hr_text = f'''Low risk: reference
HR: {hr:.2f} (95%CI {hr95_low:.2f}-{hr95_high:.2f})
Log-rank test p: {"< 0.001" if p_value < 0.001 else f"{p_value:.3f}" if p_value < 0.01 else f"{p_value:.2f}"}'''
plt666.text(2, 0.05, hr_text, fontsize=22, bbox=dict(facecolor='white', alpha=0.5),fontproperties=prop)
plt666.tight_layout()
#plt666.savefig(f'{goal} KM {modality}_{embedding}.png', dpi=600)


##scattertxt vis
import scattertext as st
from matplotlib.colors import ListedColormap

def remove_pure_numbers(text):
    # Remove standalone numbers
    text_without_numbers = re.sub(r'\b\d+\b', '', text)
    # Clean up any extra spaces
    text_without_numbers = re.sub(r'\s+', ' ', text_without_numbers).strip()
    return text_without_numbers


#vis_df = test_df[['Radiology', 'group']]
vis_df['Radiology'] = vis_df['Radiology'].apply(lambda x: (' ').join(x.split('</s>')[:3]))
vis_df['Radiology'] = vis_df['Radiology'].apply(lambda x: x.replace('anuscarcinoom', ''))
vis_df['Radiology'] = vis_df['Radiology'].apply(lambda x: x.replace('verslag revisie', ''))
vis_df['Radiology'] = vis_df['Radiology'].apply(lambda x: x.replace('d.', ''))
vis_df['Radiology'] = vis_df['Radiology'].apply(lambda x: remove_pure_numbers(x))
print(np.unique(list(vis_df['group'])))
# Step 1: Build a Corpus from the DataFrame
custom_cmap = ListedColormap(['#FF5733', '#007BFF'])
corpus = st.CorpusFromPandas(
   vis_df,
    category_col='group',  # Column with risk groups
    text_col='Radiology',            # Column with free-text data
    nlp=st.whitespace_nlp_with_sentences
).build()

# Step 2: Generate the Scattertext Plot
html = st.produce_scattertext_explorer(
    corpus,
    category='low',  # Specify the category of interest (e.g., 'high')
    category_name='Low Risk',
    not_category_name='High Risk',
    width_in_pixels=1000,
    metadata=vis_df['group'],
    show_characteristic=False,
)


