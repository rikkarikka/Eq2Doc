import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import GlobalAttention as GA
from get_config import CUDA_ON,bsz

class network_2(nn.Module):
  def __init__(self,eq_v_size,out_v_size,emb_size=100,h_size=100):
    super(network_2, self).__init__()
    self.v_size = eq_v_size
    self.h_size = h_size
    self.out_v = out_v_size
    self.emb_size = emb_size
    self.somax = nn.Softmax()
    self.emb = nn.Embedding(self.v_size+1,emb_size,padding_idx=0)
    self.dpemb = nn.Embedding(self.out_v+1,h_size//2,padding_idx=0)
    self.brnn = nn.LSTM(emb_size,h_size//2,batch_first=True,bidirectional=True)
    self.attention = GA.GlobalAttention(h_size)
    self.dp_rnn = nn.LSTMCell(h_size//2,h_size//2)

    self.pc_emb = nn.Embedding(10,h_size//2)
    self.r_emb = nn.Embedding(5,h_size//2)
    self.upscale = nn.Linear(h_size//2,h_size)
    self.toVocab = nn.Linear(h_size,self.out_v+1)
    '''
    self.down_rnn = []
    for j in range(10):
      if CUDA_ON:
        self.down_rnn.append(nn.LSTMCell(h_size//2,h_size//2).cuda())
      else:
        self.down_rnn.append(nn.LSTMCell(h_size//2,h_size//2))
    '''
    self.bsz = 0
    self.gen = False

  def set_gen(self):
    self.gen = True

  def unset_gen(self):
    self.gen = False

  def init_hidden(self,bsz):
    self.bsz = bsz
    if CUDA_ON:
      h = Variable(torch.cuda.FloatTensor(2,bsz,self.h_size//2).zero_(),requires_grad=False)
      c = Variable(torch.cuda.FloatTensor(2,bsz,self.h_size//2).zero_(),requires_grad=False)
    else:
      h = Variable(torch.Tensor(2,bsz,self.h_size//2).zero_(),requires_grad=False)
      c = Variable(torch.Tensor(2,bsz,self.h_size//2).zero_(),requires_grad=False)
    return h,c

  def init_hidden_2(self,bsz):
    self.bsz = bsz
    if CUDA_ON:
      h = Variable(torch.cuda.FloatTensor(bsz,self.h_size//2).zero_(),requires_grad=False)
      c = Variable(torch.cuda.FloatTensor(bsz,self.h_size//2).zero_(),requires_grad=False)
    else:
      h = Variable(torch.Tensor(bsz,self.h_size//2).zero_(),requires_grad=False)
      c = Variable(torch.Tensor(bsz,self.h_size//2).zero_(),requires_grad=False)
    return h,c

  def forward(self,x,dps,hc,hc2):
    dp_emb = self.dpemb(dps)

    
    x2 = self.emb(x)
    o,_ = self.brnn(x2,hc)

    outputs = []
    attns = []
    phs = 0
    hprime, cprime = hc2
    '''
    hdp = []
    cdp = []
    for j in range(10):
      hdprime, cdprime = self.init_hidden_2(self.bsz)
      hdp.append(hdprime)
      cdp.append(cdprime)
    if CUDA_ON:
      prev = [Variable(torch.cuda.FloatTensor(self.bsz,self.h_size//2).zero_()).cuda()]*10
    else:
      prev = [Variable(torch.FloatTensor(self.bsz,self.h_size//2).zero_())]*10
    '''


    for i in range(5):
      for j in range(10):
        if self.gen:
          hprime, cprime = self.dp_rnn(dp_emb,(hprime,cprime))
        else:
          hprime, cprime = self.dp_rnn(dp_emb[:,phs,:],(hprime,cprime))

        #hdp[j],cdp[j] = self.down_rnn[j](prev[j],(hdp[j],cdp[j]))

        if CUDA_ON:
          pc = self.pc_emb(Variable(torch.cuda.LongTensor(self.bsz).fill_(j)))
        else:
          pc = self.pc_emb(Variable(torch.LongTensor(self.bsz).fill_(j)))
        p = torch.cat((hprime,pc),1)
        context,attn = self.attention(p,o)
        op = self.toVocab(context)
        if self.gen:
          dp_emb = self.somax(op).data.cpu().numpy().argmax()
          if CUDA_ON:
            dp_emb = Variable(torch.cuda.LongTensor(1).fill_(int(dp_emb)))
          else:
            dp_emb = Variable(torch.LongTensor(1).fill_(int(dp_emb)))
          outputs.append(dp_emb)
          dp_emb = self.dpemb(dp_emb)
          #prev[j] = dp_emb
        else:
          outputs.append(op)
          #prev[j] = dp_emb[:,phs,:]

        attns.append(attn)
        phs +=1

    return outputs,attns

