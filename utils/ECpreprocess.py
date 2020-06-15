import pickle
import pandas as pd
import numpy as np
import os
import re

trial_list = pickle.load(open('../trial_data/trial_list','rb'))
err_list = []

for each_trial in trial_list:
  cur_split = re.split('For inclusion in the study patients should fulfil the following criteria:|For inclusion in the study patient should fulfil the following criteria:|will be included in the study:|Inclusion:|Inclusion Criteria|Inclusion criteria|INCLUSION CRITERIA|inclusion Criteria|inclusion criteria',each_trial['criteria'],maxsplit=1)
  
  cur_split = cur_split[-1]
  cur_split = re.split('exclusion criteria|will be excluded from participation in the\n        study:|Exclusion:|Exclusion Criteria|Exclusion criteria|EXCLUSION CRITERIA|exclusion Criteria',cur_split,maxsplit=1)
  cur_inclusion = cur_split[0]
  if len(cur_split) != 2:
    cur_exclusion = ""
  else:
    cur_exclusion = cur_split[1]
    cur_exclusion = re.split('For inclusion in the study patients should fulfil the following criteria:|For inclusion in the study patient should fulfil the following criteria:|will be included in the study:|Inclusion:|Inclusion Criteria|Inclusion criteria|INCLUSION CRITERIA|inclusion Criteria|inclusion criteria',cur_exclusion,maxsplit=1)[0]
  
  each_trial['inclusion'] = cur_inclusion
  each_trial['exclusion'] = cur_exclusion
  
  
all_alphabet = '1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

for each_trial in trial_list:
  #Find first Alphabet
  for i in range(len(each_trial['inclusion'])):
    if each_trial['inclusion'][i] in all_alphabet:
      break
  each_trial['inclusion'] = each_trial['inclusion'][i:]
  
  for i in range(len(each_trial['exclusion'])):
    if each_trial['exclusion'][i] in all_alphabet:
      break
  each_trial['exclusion'] = each_trial['exclusion'][i:]

#Format inclusion text to list
for each_trial in trial_list:
  if '\n\n' not in each_trial['inclusion']:
    each_trial['inclusion_list'] = [each_trial['inclusion'].replace('\n',' ')]
  else:
    each_trial['inclusion_list'] = each_trial['inclusion'].split('\n\n')
    for i in range(len(each_trial['inclusion_list'])):
      each_trial['inclusion_list'][i] = each_trial['inclusion_list'][i].replace('\n',' ')
      for j in range(len(each_trial['inclusion_list'][i])):
        if each_trial['inclusion_list'][i][j] in alphabet:
          break
      each_trial['inclusion_list'][i] = each_trial['inclusion_list'][i][j:]
  if '' in each_trial['inclusion_list']:
    each_trial['inclusion_list'].remove('')
  if ' ' in each_trial['inclusion_list']:
    each_trial['inclusion_list'].remove(' ')
    
for each_trial in trial_list:
  for each_item in each_trial['inclusion_list']:
    if len(each_item) < 10:
      if 'Niacin' not in each_item and 'Ezetemibe' not in each_item and 'warfarin' not in each_item and 'pregnancy' not in each_item and 'Pregancy' not in each_item and 'sepsis' not in each_item and 'Dementia' not in each_item and 'Diabetes' not in each_item and 'phenytoin' not in each_item:
        each_trial['inclusion_list'].remove(each_item)
      
#Format exclusion text to list
for each_trial in trial_list:
  if '\n\n' not in each_trial['exclusion']:
    each_trial['exclusion_list'] = [each_trial['exclusion'].replace('\n',' ')]
    if each_trial['number'] == 'NCT02291588':
      each_trial['exclusion_list'] = each_trial['exclusion'].split(';')
    if each_trial['number'] == 'NCT01636843':
      each_trial['exclusion_list'] = each_trial['exclusion'].split('-')
  else:
    each_trial['exclusion_list'] = each_trial['exclusion'].split('\n\n')
    for i in range(len(each_trial['exclusion_list'])):
      each_trial['exclusion_list'][i] = each_trial['exclusion_list'][i].replace('\n',' ')
      for j in range(len(each_trial['exclusion_list'][i])):
        if each_trial['exclusion_list'][i][j] in alphabet:
          break
      each_trial['exclusion_list'][i] = each_trial['exclusion_list'][i][j:]
  if '' in each_trial['exclusion_list']:
    each_trial['exclusion_list'].remove('')
  if ' ' in each_trial['exclusion_list']:
    each_trial['exclusion_list'].remove(' ')
      
for each_trial in trial_list:
  for each_item in each_trial['exclusion_list']:
    if len(each_item) < 10:
      if 'Niacin' not in each_item and 'Ezetemibe' not in each_item and 'warfarin' not in each_item and 'pregnancy' not in each_item and 'Pregancy' not in each_item and 'sepsis' not in each_item and 'Dementia' not in each_item and 'Diabetes' not in each_item and 'phenytoin' not in each_item:
        each_trial['exclusion_list'].remove(each_item)
        
pickle.dump(trial_list,open('../clean_list','wb'))