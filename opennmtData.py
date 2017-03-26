import sys
import pickle
from random import shuffle

#usage python3 anonPickle2OpenNMT.py <picklefile> <outputDir> <doc/sent> [-notest]

try:
  pfil = sys.argv[1]
  docsent = sys.argv[3]
  assert(docsent in ['doc','sent'])
  odir = sys.argv[2]#+"_"+docsent
except:
  print("usage python3 opennmtData.py <picklefile> <outputDir> <doc/sent> [-notest]")
  exit()

try:
  if sys.argv[4] == '-notest':
    TEST = False
except:
  TEST = True


def src_preds(src):
  return [" EOP ".join([" ".join(x) for x in y]) for y in src]

def src_str(src):
  return " EOS ".join([" EOP ".join([" ".join(x) for x in y]) for y in src])

def targ_str(targ):
  return " ".join(targ)

def tdt_split(x,p):
  fn = p.split("/")[-1].split(".")[0]
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
    if TEST:
      tsize = devsize*2
      test = x[devsize:tsize]
    else:
      tsize = devsize
    train = x[tsize:]
    with open("pickles/"+fn+"_split.pickle",'wb') as f:
      pickle.dump((train,dev,test),f)
  return train, dev, test

def main():
  with open(pfil,'rb') as f:
    fns,srcs,targs = pickle.load(f)

  assert(len(fns)==len(srcs))
  assert(len(srcs)==len(targs))
  x = list(range(len(fns)))
  train,dev,test = tdt_split(x,pfil)
  if docsent == 'doc':
    with open(odir+"/src-train.txt", 'w') as f:
      with open(odir+"/tgt-train.txt",'w') as g:
        with open(odir+"/train_fn",'w') as h:
          for i in train:
            s = src_str(srcs[i])+"\n"
            t = targ_str(targs[i])+"\n"
            f.write(s)
            g.write(t)
            h.write(fns[i]+"\n")

    with open(odir+"/src-val.txt", 'w') as f:
      with open(odir+"/tgt-val.txt",'w') as g:
        with open(odir+"/val_fn",'w') as h:
          for i in dev:
            s = src_str(srcs[i])+"\n"
            t = targ_str(targs[i])+"\n"
            f.write(s)
            g.write(t)
            h.write(fns[i]+"\n")

    if TEST:
      with open(odir+"/src-test.txt", 'w') as f:
        with open(odir+"/tgt-test.txt",'w') as g:
          with open(odir+"/test_fn",'w') as h:
            for i in test:
              s = src_str(srcs[i])+"\n"
              t = targ_str(targs[i])+"\n"
              f.write(s)
              g.write(t)
              h.write(fns[i]+"\n")

  elif docsent == 'sent':
    with open(odir+"/src-train.txt", 'w') as f:
      with open(odir+"/tgt-train.txt",'w') as g:
        with open(odir+"/train_fn",'w') as h:
          for i in train:
            s = src_preds(srcs[i])
            t = targs[i]
            if len(t)!=len(s):
              print("Skipping ",fns[i])
            for j in range(len(s)):
              f.write(s[j]+'\n')
              g.write(t[j]+'\n')
              h.write(fns[i]+"\n")
    with open(odir+"/src-val.txt", 'w') as f:
      with open(odir+"/tgt-val.txt",'w') as g:
        with open(odir+"/val_fn",'w') as h:
          for i in dev:
            s = src_preds(srcs[i])
            t = targs[i]
            if len(t)!=len(s):
              print("Skipping ",fns[i])
            for j in range(len(s)):
              f.write(s[j]+'\n')
              g.write(t[j]+'\n')
              h.write(fns[i]+"\n")
    if TEST:
      with open(odir+"/src-test.txt", 'w') as f:
        with open(odir+"/tgt-test.txt",'w') as g:
          with open(odir+"/test_fn",'w') as h:
            for i in test:
              s = src_preds(srcs[i])
              t = targs[i]
              if len(t)!=len(s):
                print("Skipping ",fns[i])
              for j in range(len(s)):
                f.write(s[j]+'\n')
                g.write(t[j]+'\n')
                h.write(fns[i]+"\n")

if __name__=="__main__":
  main()

