import torch
import opts
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from random import shuffle
from collections import Counter

class net(nn.Module):
  def __init__(self):
    super(net, self).__init__()    
    self.emb = nn.Embedding(opts.vocab_size, opts.h_size)
    self.brnn = nn.LSTM(opts.h_size,opts.h_size,batch_first=True,bidirectional=True)
    self.bsz = opts.bsz

  def init_hidden(self,bsz=None):
    if not bsz:
      bsz = self.bsz
    if opts.cuda:
      h = Variable(torch.cuda.FloatTensor(2,bsz,opts.h_size).zero_(),requires_grad=False)
      c = Variable(torch.cuda.FloatTensor(2,bsz,opts.h_size).zero_(),requires_grad=False)
    else:
      h = Variable(torch.Tensor(2,bsz,opts.h_size).zero_(),requires_grad=False)
      c = Variable(torch.Tensor(2,bsz,opts.h_size).zero_(),requires_grad=False)
    return h,c

  def forward(self,inp,hc):
    inp_emb = self.emb(inp)
    _,(h,c) = self.brnn(inp_emb,hc)
    return h,c
    
def vocab(l,trim=True):
  l = [x for y in l for x in y]
  ctr = Counter(l)
  if trim:
    vocab = [k for k,v in ctr.items() if v>1]
  else:
    vocab = list(ctr.keys())
  return ["<NULL>"] + vocab

def vocabize(item,vocab,seqlen=None,pad=0):
  oov = len(vocab)
  l = [vocab.index(x) if x in vocab else oov for x in item]
  if seqlen:
    l = l[:seqlen]
    l = l + [0]*(seqlen-len(l))
  return l

def load_data():
  train_h, train_c = torch.load(opts.tgt_train_data)
  val_h, val_c = torch.load(opts.tgt_val_data)
  with open(opts.eq_train_data) as f:
    _train_eq = [x.strip() for x in f.readlines()]
  with open(opts.eq_val_data) as f:
    _val_eq = [x.strip() for x in f.readlines()]
  opts.vocab = vocab(_train_eq)
  train_eq = [vocabize(x,opts.vocab,opts.maxseq) for x in _train_eq]
  val_eq = [vocabize(x,opts.vocab,opts.maxseq) for x in _val_eq]


  print(len(train_eq),train_h.size())
  assert(len(train_eq)==train_h.size(1))
  assert(len(val_eq)==val_h.size(1))

  return train_h,train_c,val_h,val_c,train_eq,val_eq

  
def make_batches(idxs):
  batches = []
  shuffle(idxs)
  i = 0
  while i<len(idxs):
    batches.append(idxs[i:i+opts.bsz])
    i+= opts.bsz
  return batches

def _fix_enc_hidden(h):
  #  the encoder hidden is  directions x batch x dim
  #  we need to convert it to batch x (directions*dim)
  return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
          .transpose(1, 2).contiguous() \
          .view(h.size(0) // 2, h.size(1), h.size(2) * 2).squeeze(0)


def validate(m,val_h,val_c,val_src,criterion):
  m.eval()
  b_src = Variable(torch.cuda.LongTensor(val_src))
  hc = m.init_hidden(len(val_src))
  b_h = Variable(_fix_enc_hidden(val_h))
  b_c = Variable(_fix_enc_hidden(val_c))
  ys = Variable(torch.cuda.LongTensor([1]*len(val_src)))
  h,c = m(b_src,hc)
  h = _fix_enc_hidden(h)
  c = _fix_enc_hidden(c)
  loss = 0
  loss += criterion(h,b_h,ys)
  loss += criterion(c,b_c,ys)
  print("Validation Loss : ",loss.data[0])
  m.train()



def main():
  # setup things
  m = net()
  criterion = nn.CosineEmbeddingLoss()
  optimizer = optim.Adam(m.parameters())
  train_h,train_c,val_h,val_c,train_src,val_src = load_data()
  batches = make_batches(list(range(len(train_src))))
  if opts.cuda:
    m.cuda()
    criterion.cuda()


  #train
  for i in range(opts.epochs):
    shuffle(batches)
    bloss = []
    for b in batches:
      hc = m.init_hidden(len(b))
      b_src = Variable(torch.cuda.LongTensor([train_src[j] for j in b]))
      _b_h = torch.stack([train_h[:,j,:] for j in b],1)
      b_h = Variable(_fix_enc_hidden(_b_h))
      _b_c = torch.stack([train_c[:,j,:] for j in b],1)
      b_c = Variable(_fix_enc_hidden(_b_c))
      ys = Variable(torch.cuda.LongTensor([1]*len(b)))

      def closure():
        optimizer.zero_grad()
        h,c = m(b_src,hc)
        h = _fix_enc_hidden(h)
        c = _fix_enc_hidden(c)
        loss = 0
        loss += criterion(h,b_h,ys)
        loss += criterion(c,b_c,ys)
        loss.backward()
        return loss

      x = optimizer.step(closure)
      bloss.append(x.data[0])
    print("Epoch "+str(i)+" Loss: ", sum(bloss)/len(bloss))

    #validate
    if i%opts.report==opts.report-1:
      validate(m,val_h,val_c,val_src,criterion)
      




if __name__=="__main__":
  main()
  
