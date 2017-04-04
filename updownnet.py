import sys
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from collections import Counter
from random import shuffle
from get_config import CUDA_ON,bsz,EPOCHS

import GlobalAttention as GA
from get_config import CUDA_ON,bsz
import plan_opts as opt

class network(nnModule):
  def __init__(self):
    super().__init__()
    self.somax = nn.Softmax()
    self.emb = nn.Embedding(opt.vocab_size,opt.emb_size,padding_idx=0)
    self.plan_emb = nn.Embedding(opt.out_vocab_size,opt.h_size,padding_idx=0)
    self.plan_rnn = nn.LSTM(opt.h_size*2,opt.h_size,opt.layers,batch_first=True)
    self.brnn = nn.LSTM(opt.emb_size,opt.h_size//2,batch_first=True,bidirectional=True)
    self.labels =["PRED","ARG0","ARG1","ARG2","ARG3","DIS","NEG","MATH0","MATH1","SENT"]
    
  def forward(self,eq):
    eq_emb = self.emb(eq)
    eq_out, eq_context = self.brnn(eq_emb)

    label_emb = self.plan_emb(self.labels)
    outputs = []
    for i in range(10):
      output, hidden = self.plan_rnn(label_emb[:,i,:])

      for j in range(5):
        emb_t = torch.cat([eq_emb[(i*5)+j],output],1)
        output, hidden = self.plan_rnn(emb_t,hidden)
        output, attn = self.attn(output, eq_context)
        output = self.dropout(output)
        outputs.append(output)

    return outputs

def tdt_split(x,fn="eq2planData"):
  try:
    with open("pickles/"+fn+"_split.pickle",'rb') as f:
      train, dev, test = pickle.load(f)
    print("loaded train dev test")
  except:
    print("Creating Train Dev Test split")
    shuffle(x)
    devsize = len(x)//10
    print(devsize)
    dev = x[:devsize]
    tsize = devsize*2
    test = x[devsize:tsize]
    train = x[tsize:]
    with open("pickles/"+fn+"_split.pickle",'wb') as f:
      pickle.dump((train,dev,test),f)
  return train, dev, test


def vocab(l,trim=True):
  l = [x for y in l for x in y]
  ctr = Counter(l)
  if trim:
    vocab = [k for k,v in ctr.items() if v>1]
  else:
    vocab = list(ctr.keys())
  return ["<NULL>"] + vocab

def dp_vocab_2(dps):
  vocab = []
  for dp in dps:
    for s in dp:
      for p in s:
        vocab.extend(p)
  v = ["<NULL>"] + [k for k,v in Counter(vocab).items() if v>1] + ["NEW_S","SAME_S","PRED","ARG0",
                                                                    "ARG1","ARG2","ARG3","ARG4",
                                                                    "DIS","NEG","MATH0","MATH1","SENT"]
  return v

def vocabize(item,vocab,seqlen=None,pad=0):
  oov = len(vocab)
  l = [vocab.index(x) if x in vocab else oov for x in item]
  if seqlen:
    l = l[:seqlen]
    l = l + [0]*(seqlen-len(l))
  return l

def dp_vocabize_2(item, vocab):
  vec = []
  flat_item = []
  for x in item:
    new_sent = "NEW_S"
    for y in x:
      flat_item.append(y+[new_sent])
      new_sent = "SAME_S"

  for x in flat_item[:5]:
    tvec = [vocab.index(y) if y in vocab else len(vocab) for y in x]
    vec.append(tvec)

  if len(vec)<5:
    vec = vec + [[0]*10 for i in range(5-len(vec))]

  turnedvec = []
  for i in range(10):
    for j in range(5):
      turnedvec.append(vec[j][i])
  #vec = [x for y in vec for x in y]
  return turnedvec

def make_batches(idxs):
  batches = []
  shuffle(idxs)
  i = 0
  while i<len(idxs):
    batches.append(idxs[i:i+bsz])
    i+= bsz
  return batches


def do_epoch(net,optimizer,criterion,src,tgt_flat,epoch):
  c = 0
  net.train()
  batches = make_batches(list(range(len(src))))
  for b in batches:
    c+=1
    bsz = len(b)
    bouttgt = torch.LongTensor([tgt_flat[i] for i in b])
    if CUDA_ON:
      bintgt = torch.cuda.LongTensor([[0]+tgt_flat[i][:-1] for i in b])
      bouttgt = torch.cuda.LongTensor([tgt_flat[i] for i in b])
      bsrc = torch.cuda.LongTensor([src[i] for i in b])
    else:
      bsrc = torch.LongTensor([src[i] for i in b])
      bintgt = torch.LongTensor([[0]+tgt_flat[i][:-1] for i in b])
      bouttgt = torch.LongTensor([tgt_flat[i] for i in b])

    inputs = Variable(bsrc)
    intgt = Variable(bintgt)
    outtgt = Variable(bouttgt)
    h,c = net.init_hidden(bsz)
    h2,c2 = net.init_hidden_2(bsz)
    if CUDA_ON:
      inputs.cuda()
      intgt.cuda()
      outtgt.cuda()
    def closure():
      optimizer.zero_grad()
      outputs,_ = net(inputs,intgt,(h,c),(h2,c2))
      loss = 0
      for j in range(50):
        loss += criterion(outputs[j],outtgt[:,j])
      loss.backward()
      #print(loss.data)
      return loss
    optimizer.step(closure)
    

def val(net,src,tgt,eq_v,out_v):
  net.eval()
  hc = net.init_hidden(1)
  hc_2 = net.init_hidden_2(1)
  out_v = out_v +  ['oov']
  ev = eq_v + ['oov']
  ostr = ""
  net.set_gen()
  dps = torch.LongTensor(1).zero_()
  acc = 0
  for c,s in enumerate(src):
    if CUDA_ON:
      dps.cuda()
      outputs,_ = net(Variable(torch.cuda.LongTensor(s).view(1,-1)),Variable(dps).cuda(),hc,hc_2)
    else:
      outputs,_ = net(Variable(torch.LongTensor(s).view(1,-1)),Variable(dps),hc,hc_2)
    tstr = ""
    # measure accuracy up to 5th pred?
    acc += sum([int(outputs[i].data[0]==tgt[c][i]) for i in range(50)])/50
    for j in range(5):
      if out_v[int(outputs[(j*10)].data[0])] == "<NULL>":
        break
      tstr += ' '.join([out_v[int(outputs[(j*10)+k].data[0])] for k in range(9)])
      if out_v[int(outputs[(j*10)+9].data[0])] == "NEW_S":
        tstr += " EOS "
      else:
        tstr += " EOP "

    #print(tstr)
    ostr += tstr + '\n'
  net.unset_gen()
  return ostr,acc


def train(out_dir):
  with open("pickles/eq2planData.pickle",'rb') as f:
    fns, eqs, dps, _ = pickle.load(f)

  
  train_idx,dev_idx,_ = tdt_split(fns)
  train_eqs = [x for i,x in enumerate(eqs) if i in train_idx]
  train_dps = [x for i,x in enumerate(dps) if i in train_idx]
  dev_eqs = [eqs[i] for i in dev_idx]
  dev_dps = [dps[i] for i in dev_idx]

  eq_v = vocab(train_eqs,False)
  out_v = dp_vocab_2(train_dps)

  train_src = [vocabize(x,eq_v,15) for x in train_eqs]
  #train_tgt = [dp_vocabize(x, pred_v, math_v, arg_v ) for x in train_dps]
  train_tgt = [dp_vocabize_2(x, out_v ) for x in train_dps]
  dev_src = [vocabize(x,eq_v) for x in dev_eqs]
  dev_tgt = [dp_vocabize_2(x, out_v ) for x in dev_dps]

  criterion = nn.CrossEntropyLoss()

  net = network(len(eq_v),len(out_v))
  optimizer = optim.Adam(net.parameters())
  if CUDA_ON:
    net.cuda()
    criterion.cuda()

  for epoch in range(EPOCHS):
    do_epoch(net,optimizer,criterion,train_src,train_tgt,batches,epoch)

    if epoch % 10 == 9:
      print("Writing %d checkpoint" % epoch)
      checkpoint = {
        'model': net,
        'vocabs': (eq_v,out_v),
        'vecs': (train_src,train_tgt,dev_src,dev_tgt),
        'epoch':epoch
      }
      torch.save(checkpoint, out_dir+"/"+str(epoch)+'checkpoint.mod')
      ostr,acc = val(net,dev_src,dev_tgt,eq_v,out_v)
      print("Valid Accuracy: %f" %acc)
      with open(out_dir+"/val-"+str(acc)+"-e"+str(epoch)+".out",'w') as f:
        f.write(ostr)

def usage():
  print("Usage: babababa")
  exit()

if __name__=="__main__":

  if sys.argv[1] == "train":
    try:
      train(sys.argv[2])
    except:
      usage()

  elif sys.argv[1] == "val":
    with open(sys.argv[2],'rb') as f:
      chpt = torch.load(f)
    net = chpt["model"]
    eq_v,out_v = chpt["vocabs"]
    train_src,train_tgt,dev_src,dev_tgt = chpt["vecs"]
    ostr = val(net,dev_src,dev_tgt,eq_v,out_v)
    with open("last-val.out",'w') as f:
      f.write(ostr)

