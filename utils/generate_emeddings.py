import torch
import numpy as np
import random
import json
import logging
import os
import pickle
from tqdm import tqdm
import pytorch_pretrained_bert
from pytorch_pretrained_bert import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertModel, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

SEED = 12345
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def convert_examples_to_features(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    return input_ids,input_mask

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

model_loc = './pretrained_bert_tf/biobert_pretrain_output_all_notes_150000/'
tokenizer = BertTokenizer.from_pretrained(model_loc, do_lower_case=True)
# Prepare model
cache_dir = os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, 'distributed_{}'.format(local_rank))
model = BertModel.from_pretrained(model_loc,
          cache_dir=cache_dir)
if fp16:
    model.half()
model.to(device)
if local_rank != -1:
    try:
        from apex.parallel import DistributedDataParallel as DDP
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

    model = DDP(model)
elif n_gpu > 1:
    model = torch.nn.DataParallel(model)

trial_list = pickle.load(open('../clean_list','rb'))

model.eval()
embedding_dict = {}
for i in tqdm(range(len(trial_list))):
  cur_dict = {}
  cur_dict['condition'] = trial_list[i]['condition']
  cur_dict['phase'] = trial_list[i]['phase']
  cur_list = trial_list[i]['inclusion_list']
  cur_inc = []
  cur_incw = []
  
  for j in range(len(cur_list)):
    cur_id, cur_mask = convert_examples_to_features(cur_list[j], tokenizer)
    cur_id = torch.tensor([cur_id], dtype=torch.long).to(device)
    cur_mask = torch.tensor([cur_mask], dtype=torch.long).to(device)
    encoder_opt, pooled_opt = model(cur_id, None, cur_mask, output_all_encoded_layers=False)
    cur_inc.append(encoder_opt.squeeze(0).detach().cpu().numpy())
    cur_incw.append(pooled_opt.squeeze(0).detach().cpu().numpy())
  cur_dict['inclusion'] = cur_inc
  cur_dict['inclusion_sentence'] = cur_incw
  
  cur_list = trial_list[i]['exclusion_list']
  cur_exc = []
  cur_excw = []
  
  for j in range(len(cur_list)):
    cur_id, cur_mask = convert_examples_to_features(cur_list[j], tokenizer)
    cur_id = torch.tensor([cur_id], dtype=torch.long).to(device)
    cur_mask = torch.tensor([cur_mask], dtype=torch.long).to(device)
    encoder_opt, pooled_opt = model(cur_id, None, cur_mask, output_all_encoded_layers=False)
    cur_exc.append(encoder_opt.squeeze(0).detach().cpu().numpy())
    cur_excw.append(pooled_opt.squeeze(0).detach().cpu().numpy())
  cur_dict['exclusion'] = cur_exc
  cur_dict['exclusion_sentence'] = cur_excw
  embedding_dict[trial_list[i]['number']] = cur_dict
pickle.dump(embedding_dict,open('../embedding_dict','wb'))