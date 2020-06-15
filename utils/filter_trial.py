import pandas as pd
import csv
import os
import numpy as np
import xml.etree.cElementTree as ET
import pickle

path = '../trial_data/'
err_list = []
trial_list = []

for each_file in os.listdir(path):
  cur_trial = ET.ElementTree(file=path+'/'+each_file)
  cur_dict = {}
  cur_dict['number'] = each_file[:-4]
  cur_dict['condition'] = cur_trial.find('condition').text
  cur_dict['criteria'] = cur_trial.find('eligibility/criteria/textblock').text
  cur_dict['gender'] = cur_trial.find('eligibility/gender').text
  cur_phase = cur_trial.find('phase')
  if cur_phase != None:
    cur_dict['phase'] = cur_phase.text
  else:
    cur_dict['phase'] = None
  cur_dict['min_age'] = cur_trial.find('eligibility/minimum_age').text
  cur_dict['max_age'] = cur_trial.find('eligibility/maximum_age').text
  trial_list.append(cur_dict)
  
for each_trial in trial_list:
  cur_age = each_trial['max_age']
  for i in range(len(cur_age)):
    if cur_age[i] == 'N':
      new_age = np.nan
      break
    elif cur_age[i] == ' ':
      new_age = int(cur_age[:i])
      break
  each_trial['max_age'] = new_age
  
  cur_age = each_trial['min_age']
  for i in range(len(cur_age)):
    if cur_age[i] == 'N':
      new_age = np.nan
      break
    elif cur_age[i] == ' ':
      new_age = int(cur_age[:i])
      break
  each_trial['min_age'] = new_age
  
  if each_trial['gender'] == 'Male':
    new_gender = 1
  elif each_trial['gender'] == 'Female':
    new_gender = 0
  elif each_trial['gender'] == 'All':
    new_gender = 2
  each_trial['gender'] = new_gender
    
pickle.dump(trial_list,open('../trial_list','wb'))