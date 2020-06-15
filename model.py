#Pool and classification
import torch
from torch import nn
import torch.nn.functional as F
import random
 
class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)
        
        self.__padding = (kernel_size - 1) * dilation
        
    def forward(self, input):
        return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))

class HighwayBlock(nn.Module):
  def __init__(self, input_dim, kernel_size):
    super(HighwayBlock, self).__init__()
    self.conv_t = CausalConv1d(input_dim, input_dim, kernel_size)
    self.conv_z = CausalConv1d(input_dim, input_dim, kernel_size)
    
  def forward(self, input):
    t = torch.sigmoid(self.conv_t(input))
    z = t * self.conv_z(input) + (1-t) * input
    return z
    
class ECEmbedding(nn.Module):
  def __init__(self, word_dim, conv_dim, mem_dim):
    super(ECEmbedding, self).__init__()
    #Input: batch_size * sentence_len * embd_dim
    self.word_dim = word_dim
    self.conv_dim = conv_dim
    
    
    #High-way network for word features
    self.init_conv1 = CausalConv1d(word_dim, conv_dim, 1)
    self.init_conv2 = CausalConv1d(word_dim, conv_dim, 3)
    self.init_conv3 = CausalConv1d(word_dim, conv_dim, 5)
    self.init_conv4 = CausalConv1d(word_dim, conv_dim, 7)
    self.highway1 = HighwayBlock(4*conv_dim, 3)
    self.highway2 = HighwayBlock(4*conv_dim, 3)
    self.highway3 = HighwayBlock(4*conv_dim, 3)

    self.pool = nn.AdaptiveMaxPool1d(1)
    
  def forward(self, input, mask):
    #Input: B * L * embd_dim
    #Mask: B * L
    input = input.permute(0,2,1)
    conv1 = self.init_conv1(input)
    conv2 = self.init_conv2(input)
    conv3 = self.init_conv3(input)
    conv4 = self.init_conv4(input)
    concat = torch.cat((conv1,conv2,conv3,conv4), dim=1)

    highway_res = self.highway1(concat)
    highway_res = torch.relu(highway_res)

    highway_res = self.highway2(highway_res)
    highway_res = torch.relu(highway_res)
    highway_res = self.highway3(highway_res)
    highway_res = torch.relu(highway_res)

    highway_res = highway_res * mask.unsqueeze(1)
    pooled_res = self.pool(highway_res)
    pooled_res = pooled_res.squeeze(-1)
    return pooled_res
  
class EHRMemoryNetwork(nn.Module):
  def __init__(self, word_dim, mem_dim, demo_dim):
    super(EHRMemoryNetwork, self).__init__()
    self.mem_dim = mem_dim
    
    self.erase_layer = nn.Linear(word_dim, mem_dim)
    self.add_layer = nn.Linear(word_dim, mem_dim)
    self.demo_embd = nn.Linear(demo_dim, mem_dim)
    
    self.init_memory = nn.Parameter(torch.randn(12, mem_dim))
    
  def forward(self, input, demo, mask):    
    batch_size = input.size(0)
    time_step = input.size(1)
    assert input.size(2) == 12
    word_dim = input.size(3)
    
    memory = self.init_memory.unsqueeze(0).repeat(batch_size,1,1)
    demo_mem = torch.tanh(self.demo_embd(demo))
    
    for i in range(time_step):
      cur_input = input[:, i, :, :].reshape(batch_size*12, word_dim)
      erase = torch.sigmoid(self.erase_layer(cur_input))
      add = torch.tanh(self.add_layer(cur_input))
      erase = erase.reshape(batch_size, 12, self.mem_dim)
      add = add.reshape(batch_size, 12, self.mem_dim)
      cur_mask = mask[:, i].reshape(batch_size, 1, 1)
      erase = erase * cur_mask
      add = add * cur_mask
      memory = memory * (1 - erase) + add
    
    memory = torch.cat((memory, demo_mem.unsqueeze(1)), dim=1)
    return memory

class QueryNetwork(nn.Module):
  def __init__(self, mem_dim, conv_dim, mlp_dim):
    super(QueryNetwork, self).__init__()
    self.word_trans = nn.Linear(4*conv_dim,mem_dim, bias=False)
    self.mlp = nn.Linear(2*mem_dim, mlp_dim)
    self.output = nn.Linear(mlp_dim, 3)
  
  def forward(self, memory, query):
    #query: bs, 4*conv_dim
    #memory: bs, 13, mem_dim
    trans_query = self.word_trans(query) #bs, mem
    trans_query = torch.relu(trans_query)
    attention = torch.bmm(trans_query.unsqueeze(1), memory.permute(0,2,1)).squeeze(1) #B*13
    attention = torch.softmax(attention, dim=-1)
    response = attention.unsqueeze(-1) * memory #B*13*m
    response = torch.mean(response, dim=1, keepdim=False) #B*m
    
    output = torch.cat((response, trans_query), dim=-1)
    output = self.mlp(output)
    output = torch.relu(output)
    output = self.output(output)
    return output, response, trans_query, attention

def get_loss(criteria, criteria_mask, 
             ehr, ehr_mask, demo, label,
             query_network, ehr_network, ec_network, device):
  
  memory = ehr_network(ehr, demo, ehr_mask) # batch_size, class_num
  criteria_embd = ec_network(criteria, criteria_mask) #ec_num, mem_dim

  similarity_label = []
  label_mask = []
  for i in range(len(label)):
    if label[i] == 0:
      similarity_label.append(1)
      label_mask.append(1)
    elif label[i] == 1:
      similarity_label.append(-1)
      label_mask.append(1)
    elif label[i] == 2:
      similarity_label.append(1)
      label_mask.append(0)
      
  similarity_label = torch.tensor(similarity_label, dtype=torch.long).to(device)
  label_mask = torch.tensor(label_mask, dtype=torch.float32).to(device)
  
  ce_loss = nn.CrossEntropyLoss()
  sm_loss = nn.CosineEmbeddingLoss(margin=0.3, reduction='none')
  
  output, response, query, attention = query_network(memory, criteria_embd) #bs, 3
  pred = torch.softmax(output, dim=-1)
  loss = ce_loss(output, label)
  similarity = sm_loss(response, query, similarity_label)
  similarity = similarity * label_mask
  similarity = torch.sum(similarity) / torch.sum(label_mask)
  
  return loss, similarity, pred, attention, response, query