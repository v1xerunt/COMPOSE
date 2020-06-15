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
%matplotlib inline
from matplotlib import pyplot as plt
import seaborn as sns

import model4
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import manifold
from scipy.spatial import distance
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D
 
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

embedding_dict = pickle.load(open('./embedding_dict','rb'))
prod_dataset = pickle.load(open('./prod_dataset','rb'))
proc_dataset = pickle.load(open('./proc_dataset','rb'))
diag_dataset = pickle.load(open('./diag_dataset','rb'))


test_demo_dataset = pickle.load(open('./test_demo_dataset','rb'))
test_ehr_dataset = pickle.load(open('./test_ehr_dataset','rb'))
test_id_dataset = pickle.load(open('./test_id_dataset','rb'))
test_trial_dataset = pickle.load(open('./test_trial_dataset','rb'))
test_label_dataset = pickle.load(open('./test_label_dataset','rb'))

def get_batch(loc, batch_size, mode):
  if mode == 'train':
    batch_id = train_id_dataset[loc:loc+batch_size]
    ehr = np.array(train_ehr_dataset)[batch_id]
    demo = np.array(train_demo_dataset)[batch_id]
    cur_trial = train_trial_dataset[loc:loc+batch_size]
    batch_label = train_label_dataset[loc:loc+batch_size]
  elif mode == 'valid':
    batch_id = valid_id_dataset[loc:loc+batch_size]
    ehr = np.array(valid_ehr_dataset)[batch_id]
    demo = np.array(valid_demo_dataset)[batch_id]
    cur_trial = valid_trial_dataset[loc:loc+batch_size]
    batch_label = valid_label_dataset[loc:loc+batch_size]
  else:
    batch_id = test_id_dataset[loc:loc+batch_size]
    ehr = np.array(test_ehr_dataset)[batch_id]
    demo = np.array(test_demo_dataset)[batch_id]
    cur_trial = test_trial_dataset[loc:loc+batch_size]
    batch_label = test_label_dataset[loc:loc+batch_size]
  
  batch_ehr = []
  batch_demo = []
  max_ts = 0
  for each_id in range(len(batch_id)):
    if len(ehr[each_id]) > max_ts:
      max_ts = len(ehr[each_id])
    
  batch_ehr_mask = []
  for each_id in range(len(batch_id)):    
    tmp_ehr = np.zeros((max_ts, 12, word_dim))
    tmp_mask = np.zeros(max_ts)
    tmp = [np.vstack((diag_dataset[ehr[each_id][0][0]],prod_dataset[ehr[each_id][0][1]],proc_dataset[ehr[each_id][0][2]]))]
    for j in range(1, len(ehr[each_id])):
      tmp.append(np.vstack((diag_dataset[ehr[each_id][j][0]],prod_dataset[ehr[each_id][j][1]],proc_dataset[ehr[each_id][j][2]])))
    tmp = np.array(tmp)
    tmp_ehr[:tmp.shape[0], :, :] = tmp
    tmp_mask[:tmp.shape[0]] = 1
    batch_ehr.append(tmp_ehr)
    batch_ehr_mask.append(tmp_mask)
    batch_demo.append(demo[each_id])
    
  batch_criteria = []
  batch_criteria_mask = []
  max_seq = 0
  for each_id in range(len(batch_id)):
    tmp_trial, tmp_type, tmp_ec  = cur_trial[each_id]
    if tmp_type == 'i':
      if len(embedding_dict[tmp_trial]['inclusion'][tmp_ec]) > max_seq:
        max_seq = len(embedding_dict[tmp_trial]['inclusion'][tmp_ec])
    else:
      if len(embedding_dict[tmp_trial]['exclusion'][tmp_ec]) > max_seq:
        max_seq = len(embedding_dict[tmp_trial]['exclusion'][tmp_ec])
      
  for each_id in range(len(batch_id)):
    tmp_trial, tmp_type, tmp_ec  = cur_trial[each_id]
    tmp = np.zeros((max_seq, word_dim))
    tmp_mask = np.zeros((max_seq))
    if tmp_type == 'i':
      tmp[:len(embedding_dict[tmp_trial]['inclusion'][tmp_ec]), :] = embedding_dict[tmp_trial]['inclusion'][tmp_ec]
      tmp_mask[:len(embedding_dict[tmp_trial]['inclusion'][tmp_ec])] = 1 
    else:
      tmp[:len(embedding_dict[tmp_trial]['exclusion'][tmp_ec]), :] = embedding_dict[tmp_trial]['exclusion'][tmp_ec]
      tmp_mask[:len(embedding_dict[tmp_trial]['exclusion'][tmp_ec])] = 1
       
    batch_criteria.append(tmp)
    batch_criteria_mask.append(tmp_mask)

  batch_criteria = np.array(batch_criteria)
  batch_criteria_mask = np.array(batch_criteria_mask)
  batch_ehr = np.array(batch_ehr)
  batch_ehr_mask = np.array(batch_ehr_mask)
  batch_demo = np.array(batch_demo)
  batch_label = np.array(batch_label)

  return batch_ehr, batch_ehr_mask, batch_demo, batch_criteria, batch_criteria_mask, batch_label

word_dim = 768
conv_dim = 128
mem_dim = 320
mlp_dim = 512
demo_dim = 3
class_dim = 477
margin = 1

batch_size = 8
epoch = 50
lr=1e-3

embedding_network = model4.ECEmbedding(word_dim, conv_dim, mem_dim).to(device)
memory_network = model4.EHRMemoryNetwork(word_dim, mem_dim, demo_dim).to(device)
query_network = model4.QueryNetwork(mem_dim, conv_dim, mlp_dim).to(device)
optimizer = torch.optim.Adam(list(embedding_network.parameters())+list(memory_network.parameters())+list(query_network.parameters()), lr=lr)

checkpoint = torch.load('./save/model')
embedding_network.load_state_dict(checkpoint['embedding'])
memory_network.load_state_dict(checkpoint['memory'])
query_network.load_state_dict(checkpoint['query'])
optimizer.load_state_dict(checkpoint['optimizer'])

loss_list = []
pred_list = []
res_list =[]
query_list = []
true_list = []
sim_list = []
att_list = []
for iteration in range(0, len(test_trial_dataset), batch_size):
  embedding_network.eval()
  memory_network.eval()
  query_network.eval()
    
  batch_ehr, batch_ehr_mask, batch_demo, batch_criteria, batch_criteria_mask, batch_label = get_batch(iteration, batch_size, 'test')
  
  batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
  batch_ehr_mask = torch.tensor(batch_ehr_mask, dtype=torch.float32).to(device)
  batch_demo = torch.tensor(batch_demo, dtype=torch.float32).to(device)
  batch_criteria = torch.tensor(batch_criteria, dtype=torch.float32).to(device)
  batch_criteria_mask = torch.tensor(batch_criteria_mask, dtype=torch.float32).to(device)
  batch_label = torch.tensor(batch_label, dtype=torch.long).to(device)
  
  optimizer.zero_grad()

  loss, sim, pred, att, res, query = model4.get_loss(batch_criteria, batch_criteria_mask, 
            batch_ehr, batch_ehr_mask, batch_demo, batch_label,
            query_network, memory_network, embedding_network, device)
  att_list += list(att.cpu().detach().numpy())
  loss_list.append(loss.cpu().detach().numpy() if loss != 0 else 0)
  pred_list += list(pred.cpu().detach().numpy())
  true_list += list(batch_label.cpu().detach().numpy())
  res_list += list(res.cpu().detach().numpy())
  query_list += list(query.cpu().detach().numpy())
  sim_list.append(sim.cpu().detach().numpy())
  if iteration % 2000 == 0:
    print('%d / %d'%(iteration, len(test_trial_dataset)))
  
new_true = []
new_pred = []
new_round = []
for i in range(len(pred_list)):
  if true_list[i] == 0:
    new_pred.append(1-pred_list[i][0])
    new_round.append(np.round(1-pred_list[i][0]))
    new_true.append(0)
  elif true_list[i] == 1:
    new_pred.append(pred_list[i][1])
    new_round.append(np.round(pred_list[i][1]))
    new_true.append(1)
    
auroc = metrics.roc_auc_score(new_true, new_pred)
print('auroc: ', auroc)
acc = metrics.accuracy_score(new_true, new_round)
print('acc: ', acc)
(precisions, recalls, thresholds) = metrics.precision_recall_curve(new_true, new_pred)
auprc = metrics.auc(recalls, precisions)
print('prc: ', auprc)
f1 = metrics.f1_score(new_true, new_round)
print('f1: ', f1)