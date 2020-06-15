import torch
import numpy as np
import random
import json
import logging
import os
import pickle
import pandas as pd
import importlib
from tqdm import tqdm

import model2
from sklearn.model_selection import train_test_split

SEED = 12345
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

local_rank = -1
fp16 = False
if local_rank == -1:
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  n_gpu = torch.cuda.device_count()
else:
  torch.cuda.set_device(local_rank)
  device = torch.device("cuda", local_rank)
  n_gpu = 1
  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
  torch.distributed.init_process_group(backend='nccl')
logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(device, n_gpu, bool(local_rank != -1), fp16))
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(SEED)

ehr_data = pd.read_csv('../ehr_data/records.csv', dtype={'claim_id':str})
proc_dict = pickle.load(open('../proc_dict','rb'))
diag_dict = pickle.load(open('../diag_dict','rb'))
prod_dict = pickle.load(open('../prod_dict','rb'))
embedding_dict = pickle.load(open('../embedding_dict','rb'))

claim_trial_dict = {}
total_trials = []
for idx, row in ehr_data.iterrows():
  if row['claim_id'] not in claim_trial_dict:
    claim_trial_dict[row['claim_id']] = [row['nct_id']]
  elif row['nct_id'] not in claim_trial_dict[row['claim_id']]:
    claim_trial_dict[row['claim_id']].append(row['nct_id'])
  if row['nct_id'] not in total_trials:
    total_trials.append(row['nct_id'])

age_list = []
for idx, row in ehr_data.iterrows():
  if row['pat_age_yr_nbr'] > 0:
    age_list.append(row['pat_age_yr_nbr'])
mean_age = np.mean(age_list)
std_age = np.std(age_list)

valid_ehr_dataset = []
test_ehr_dataset = []
train_ehr_dataset = []

valid_trial_dataset = []
test_trial_dataset = []
train_trial_dataset = []

valid_demo_dataset = []
test_demo_dataset = []
train_demo_dataset = []

valid_label_dataset = []
test_label_dataset = []
train_label_dataset = []

valid_id_dataset = []
test_id_dataset = []
train_id_dataset = []

total_claims = list(claim_trial_dict.keys())
idx = list(range(len(total_claims)))
train_idx, test_idx = train_test_split(idx, test_size=0.3, random_state=123, shuffle=True)
train_idx, valid_idx = train_test_split(train_idx, test_size=0.1, random_state=123, shuffle=True)

diag_dataset = {}
diag_dataset['nan'] = np.zeros((4,768))
for each_diag in diag_dict:
  diag_dataset[each_diag] = np.vstack((diag_dict[each_diag]['lv1'], diag_dict[each_diag]['lv2'], diag_dict[each_diag]['lv3'], diag_dict[each_diag]['lv4']))

prod_dataset = {}
prod_dataset['nan'] = np.zeros((4,768))
for each_prod in prod_dict:
  prod_dataset[each_prod] = np.vstack((prod_dict[each_prod]['lv1'], prod_dict[each_prod]['lv2'], prod_dict[each_prod]['lv3'], prod_dict[each_prod]['lv4']))

proc_dataset = {}
proc_dataset['nan'] = np.zeros((4,768))
for each_proc in proc_dict:
  proc_dataset[each_proc] = np.vstack((proc_dict[each_proc]['lv1'], proc_dict[each_proc]['lv2'], proc_dict[each_proc]['lv3'], proc_dict[each_proc]['lv4']))
 

    
pickle.dump(diag_dataset,open('../diag_dataset', 'wb'))
pickle.dump(prod_dataset,open('../prod_dataset', 'wb'))
pickle.dump(proc_dataset,open('../proc_dataset', 'wb'))



for i in range(len(test_idx)):
  cur_idx = test_idx[i]
  cur_data = ehr_data.loc[ehr_data['claim_id'] == total_claims[cur_idx]]
  initial_flag = True
  for idx, row in cur_data.iterrows():
    if row['diag_cd'] not in diag_dict:
      cur_diag = 'nan'
    else:
      cur_diag = row['diag_cd']
      
    if row['product_id'] not in prod_dict:
      cur_prod = 'nan'
    else:
      cur_prod = row['product_id']
      
    if row['origl_prod_svc_qlfr_cd'] not in proc_dict:
      cur_proc = 'nan'
    else:
      cur_proc = row['origl_prod_svc_qlfr_cd']
     
    if initial_flag == True:
      cur_trial = [row['nct_id']]
      initial_flag = False
      cur_ehr = [[cur_diag,cur_prod,cur_proc]]
      cur_age = 0 if row['pat_age_yr_nbr'] < 0 else (row['pat_age_yr_nbr'] - mean_age) / std_age
      cur_gender = np.array([0,1]) if row['pat_gender_cd'] == 'M' else np.array([1,0])
    else:
      cur_trial.append(row['nct_id'])
      tmp_ehr = [cur_diag,cur_prod,cur_proc]
      cur_ehr.append(tmp_ehr)
  
  test_ehr_dataset.append(cur_ehr)
  test_demo_dataset.append(np.hstack((np.array(cur_age),cur_gender)))
  
  for j in range(len(cur_trial)):
    for k in range(len(embedding_dict[cur_trial[j]]['inclusion'])):
      test_trial_dataset.append((cur_trial[j], 'i', k))
      test_id_dataset.append(i)
      test_label_dataset.append(0)
      
      rand_idx = random.choice([h for h in total_trials if h not in cur_trial])
      rand_ec = random.choice([h for h in range(len(embedding_dict[rand_idx]['inclusion']))])
      test_id_dataset.append(i)
      test_trial_dataset.append((rand_idx, 'i', rand_ec))
      test_label_dataset.append(2)
    
    if len(embedding_dict[cur_trial[j]]['exclusion']) > 0:
      for k in range(len(embedding_dict[cur_trial[j]]['exclusion'])):
        test_trial_dataset.append((cur_trial[j], 'e', k))
        test_id_dataset.append(i)
        test_label_dataset.append(1)
        
        rand_idx = random.choice([h for h in total_trials if h not in cur_trial])
        if len(embedding_dict[rand_idx]['exclusion']) > 0:
          rand_ec = random.choice([h for h in range(len(embedding_dict[rand_idx]['exclusion']))])
          test_id_dataset.append(i)
          test_trial_dataset.append((rand_idx, 'e', rand_ec))
          test_label_dataset.append(2)
    
pickle.dump(test_ehr_dataset,open('../test_ehr_dataset', 'wb'))
pickle.dump(test_demo_dataset,open('../test_demo_dataset', 'wb'))
pickle.dump(test_trial_dataset,open('../test_trial_dataset', 'wb'))
pickle.dump(test_id_dataset,open('../test_id_dataset', 'wb'))
pickle.dump(test_label_dataset,open('../test_label_dataset', 'wb'))


for i in range(len(valid_idx)):
  cur_idx = valid_idx[i]
  cur_data = ehr_data.loc[ehr_data['claim_id'] == total_claims[cur_idx]]
  initial_flag = True
  for idx, row in cur_data.iterrows():
    if row['diag_cd'] not in diag_dict:
      cur_diag = 'nan'
    else:
      cur_diag = row['diag_cd']
      
    if row['product_id'] not in prod_dict:
      cur_prod = 'nan'
    else:
      cur_prod = row['product_id']
      
    if row['origl_prod_svc_qlfr_cd'] not in proc_dict:
      cur_proc = 'nan'
    else:
      cur_proc = row['origl_prod_svc_qlfr_cd']
     
    if initial_flag == True:
      cur_trial = [row['nct_id']]
      initial_flag = False
      cur_ehr = [[cur_diag,cur_prod,cur_proc]]
      cur_age = 0 if row['pat_age_yr_nbr'] < 0 else (row['pat_age_yr_nbr'] - mean_age) / std_age
      cur_gender = np.array([0,1]) if row['pat_gender_cd'] == 'M' else np.array([1,0])
    else:
      cur_trial.append(row['nct_id'])
      tmp_ehr = [cur_diag,cur_prod,cur_proc]
      cur_ehr.append(tmp_ehr)
  
  valid_ehr_dataset.append(cur_ehr)
  valid_demo_dataset.append(np.hstack((np.array(cur_age),cur_gender)))
  
  for j in range(len(cur_trial)):
    for k in range(len(embedding_dict[cur_trial[j]]['inclusion'])):
      valid_trial_dataset.append((cur_trial[j], 'i', k))
      valid_id_dataset.append(i)
      valid_label_dataset.append(0)
      
      rand_idx = random.choice([h for h in total_trials if h not in cur_trial])
      rand_ec = random.choice([h for h in range(len(embedding_dict[rand_idx]['inclusion']))])
      valid_id_dataset.append(i)
      valid_trial_dataset.append((rand_idx, 'i', rand_ec))
      valid_label_dataset.append(2)
    
    if len(embedding_dict[cur_trial[j]]['exclusion']) > 0:
      for k in range(len(embedding_dict[cur_trial[j]]['exclusion'])):
        valid_trial_dataset.append((cur_trial[j], 'e', k))
        valid_id_dataset.append(i)
        valid_label_dataset.append(1)
        
        rand_idx = random.choice([h for h in total_trials if h not in cur_trial])
        if len(embedding_dict[rand_idx]['exclusion']) > 0:
          rand_ec = random.choice([h for h in range(len(embedding_dict[rand_idx]['exclusion']))])
          valid_id_dataset.append(i)
          valid_trial_dataset.append((rand_idx, 'e', rand_ec))
          valid_label_dataset.append(2)
    
pickle.dump(valid_ehr_dataset,open('../valid_ehr_dataset', 'wb'))
pickle.dump(valid_demo_dataset,open('../valid_demo_dataset', 'wb'))
pickle.dump(valid_trial_dataset,open('../valid_trial_dataset', 'wb'))
pickle.dump(valid_id_dataset,open('../valid_id_dataset', 'wb'))
pickle.dump(valid_label_dataset,open('../valid_label_dataset', 'wb'))



for i in range(len(train_idx)):
  cur_idx = train_idx[i]
  cur_data = ehr_data.loc[ehr_data['claim_id'] == total_claims[cur_idx]]
  initial_flag = True
  for idx, row in cur_data.iterrows():
    if row['diag_cd'] not in diag_dict:
      cur_diag = 'nan'
    else:
      cur_diag = row['diag_cd']
      
    if row['product_id'] not in prod_dict:
      cur_prod = 'nan'
    else:
      cur_prod = row['product_id']
      
    if row['origl_prod_svc_qlfr_cd'] not in proc_dict:
      cur_proc = 'nan'
    else:
      cur_proc = row['origl_prod_svc_qlfr_cd']
     
    if initial_flag == True:
      cur_trial = [row['nct_id']]
      initial_flag = False
      cur_ehr = [[cur_diag,cur_prod,cur_proc]]
      cur_age = 0 if row['pat_age_yr_nbr'] < 0 else (row['pat_age_yr_nbr'] - mean_age) / std_age
      cur_gender = np.array([0,1]) if row['pat_gender_cd'] == 'M' else np.array([1,0])
    else:
      cur_trial.append(row['nct_id'])
      tmp_ehr = [cur_diag,cur_prod,cur_proc]
      cur_ehr.append(tmp_ehr)
  
  train_ehr_dataset.append(cur_ehr)
  train_demo_dataset.append(np.hstack((np.array(cur_age),cur_gender)))
  
  for j in range(len(cur_trial)):
    for k in range(len(embedding_dict[cur_trial[j]]['inclusion'])):
      train_trial_dataset.append((cur_trial[j], 'i', k))
      train_id_dataset.append(i)
      train_label_dataset.append(0)
      
      rand_idx = random.choice([h for h in total_trials if h not in cur_trial])
      rand_ec = random.choice([h for h in range(len(embedding_dict[rand_idx]['inclusion']))])
      train_id_dataset.append(i)
      train_trial_dataset.append((rand_idx, 'i', rand_ec))
      train_label_dataset.append(2)
    
    if len(embedding_dict[cur_trial[j]]['exclusion']) > 0:
      for k in range(len(embedding_dict[cur_trial[j]]['exclusion'])):
        train_trial_dataset.append((cur_trial[j], 'e', k))
        train_id_dataset.append(i)
        train_label_dataset.append(1)
        
        rand_idx = random.choice([h for h in total_trials if h not in cur_trial])
        if len(embedding_dict[rand_idx]['exclusion']) > 0:
          rand_ec = random.choice([h for h in range(len(embedding_dict[rand_idx]['exclusion']))])
          train_id_dataset.append(i)
          train_trial_dataset.append((rand_idx, 'e', rand_ec))
          train_label_dataset.append(2)
    
pickle.dump(train_ehr_dataset,open('../train_ehr_dataset', 'wb'))
pickle.dump(train_demo_dataset,open('../train_demo_dataset', 'wb'))
pickle.dump(train_trial_dataset,open('../train_trial_dataset', 'wb'))
pickle.dump(train_id_dataset,open('../train_id_dataset', 'wb'))
pickle.dump(train_label_dataset,open('../train_label_dataset', 'wb'))

