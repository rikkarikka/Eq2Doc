import sys
import pickle

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
eq_v = vocab(train_eqs,False)
out_v = dp_vocab_2(train_dps)

print("VOCABS MADE")
train_src = [vocabize(x,eq_v,15) for x in train_eqs]
train_tgt = [dp_vocabize_2(x, out_v ) for x in train_dps]
print("TRAINING DATA MADE")
dev_src = [vocabize(x,eq_v) for x in dev_eqs]
dev_tgt = [dp_vocabize_2(x, out_v ) for x in dev_dps]
print("VAL DATA MADE")

pickle.dump((eq_v,out_v,train_src,train_tgt,dev_dev_tgt),open("pickles/roc_first2doc.pickle",'wb'))


