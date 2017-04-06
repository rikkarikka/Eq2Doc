import sys
import pickle
from collections import Counter



def vocab(l,trim=True):
  l = [x for y in l for x in y]
  ctr = Counter(l)
  if trim:
    vocab = [k for k,v in ctr.most_common(5000)]
  else:
    vocab = list(ctr.keys())
  v = ["<NULL>"] + vocab
  vd = {}
  dv = {}
  for i,x in enumerate(v):
    vd[x] = i
    dv[i] = x
  return vd,dv


def dp_vocab_2(dps):
  vocab = []
  for dp in dps:
    for s in dp:
      for p in s:
        vocab.extend(p)
  v = ["<NULL>"] + [k for k,v in Counter(vocab).most_common(15000)] + ["NEW_S","SAME_S"]

  vd = {}
  dv = {}
  for i,x in enumerate(v):
    vd[x] = i
    dv[i] = x
  return vd,dv

def vocabize(item,vocab,seqlen=None,pad=0):
  oov = len(vocab.keys())
  l = [vocab[x] if x in vocab else oov for x in item]
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

  oov = len(vocab.keys())

  for x in flat_item[:5]:
    tvec = [vocab[y] if y in vocab else oov for y in x]
    vec.append(tvec)

  if len(vec)<5:
    vec = vec + [[0]*10 for i in range(5-len(vec))]
  turnedvec = []
  for i in range(10):
    for j in range(5):
      turnedvec.append(vec[j][i])
  return turnedvec

print("DOING ROC CORPUS")
with open("roc/first-train.txt") as f:
  train_eqs = [x.strip().split() for x in f]
with open("roc/first-val.txt") as f:
  dev_eqs = [x.strip().split() for x in f]
with open("roc/doc-train.txt") as f:
  train_dps = [x.strip() for x in f]
  train_dps = [x.split(" EOS ") for x in train_dps]
  for i in range(len(train_dps)):
    train_dps[i] = [x.split(" EOP ") for x in train_dps[i]]
    for j in range(len(train_dps[i])):
      train_dps[i][j] = [x.split(" ") for x in train_dps[i][j]]
with open("roc/doc-val.txt") as f:
  dev_dps = [x.strip() for x in f]
  dev_dps = [x.split(" EOS ") for x in dev_dps]
  for i in range(len(dev_dps)):
    dev_dps[i] = [x.split(" EOP ") for x in dev_dps[i]]
    for j in range(len(dev_dps[i])):
      dev_dps[i][j] = [x.split(" ") for x in dev_dps[i][j]]

print("DATA LOADED")
eq_v,req = vocab(train_eqs,False)
out_v,rout = dp_vocab_2(train_dps)
pickle.dump((eq_v,req,out_v,rout),open("pickles/roc_first2doc.vocabs",'wb'))

print("VOCABS MADE")
train_src = [vocabize(x,eq_v,15) for x in train_eqs]
train_tgt = [dp_vocabize_2(x, out_v ) for x in train_dps]
print("TRAINING DATA MADE")
dev_src = [vocabize(x,eq_v) for x in dev_eqs]
dev_tgt = [dp_vocabize_2(x, out_v ) for x in dev_dps]
print("VAL DATA MADE")

pickle.dump((train_src,train_tgt,dev_src,dev_tgt),open("pickles/roc_first2doc.datapoints",'wb'))


