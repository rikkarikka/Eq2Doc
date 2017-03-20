import sys
import os
import pickle
from collections import OrderedDict

copulas = ['is','am','are','was','were','be','being','been']

def parse(fn):
  with open(fn) as f:
    data = f.read()

  sents = []
  tmp = []
  for x in data.split("\n"):
    x = x.strip()
    if not x:
      sents.append("\n".join(tmp))
      tmp = []
    else:
      tmp.append(x)

  whole_parse = []
  srl = []
  for i,s in enumerate(sents):
    t_srl = {}
    if not s or s=="NO DEPS": 
      srl.append([])
      continue
    trips = [x.split()[0:5] for x in s.split("\n")]
    whole_parse.append(trips)
    preds = set([" ".join(x[0:2]) for x in trips if "ARG" in x[2]])
    for p in preds:
      ps = p.split(" ")
      args = [x for x in trips if x[0] == ps[0] and x[1] == ps[1] and x[2] != "NONE"]
      for a in args:
        if a[2] in ["TMP","LOC"]:
          p =" ".join(a[3:5])
          aftr = [x[2:4] for x in trips if x[0] == a[0] and int(x[4]) > int(a[1])]
          if aftr:
            appnd = tuple(aftr[0])
          else:
            appnd = (a[2],a[0])
        elif "ARG" not in a[2] and a[2] != "NONE":
          appnd = (a[2],a[0])
        else:
          appnd = tuple(a[2:4])
        
        if p not in t_srl:
          t_srl[p] = []
        t_srl[p].append(appnd)

    # deal w/ copulas
    cops = [x[0:2] for x in trips if x[0] in copulas]
    for c in cops:
      deps = [x[3]+" "+x[4] for x in trips if x[0:2] == c]
      if any([x in t_srl for x in deps]):
        continue
      else:
        l = sorted([x for x in trips if x[0:2] == c],key=lambda x : x[4])
        arg = 0
        l_list = []
        for x in l:
          l_list.append(("ARG"+str(arg),x[3]))
          arg +=1
        t_srl[" ".join(c)] = l_list #[x[2:4] for x in l]


    #sort by pred order
    ordered = []
    for k in t_srl.keys():
      p , num = k.split(" ")
      ordered.append((int(num),(p,t_srl[k])))
    ordered.sort()
    ordered = [x[1] for x in ordered]
    srl.append(ordered)


  return srl

def standardize(srl,t):
  stds = []
  for p,l in srl:
    std = [p]
    l.sort()
    d = OrderedDict(l)
    if "NUMS" in d:
      std.extend(d["NUMS"])
      del(d["NUMS"])
    else:
      std.append("-")
      std.append("-")
    if "KWS" in d:
      std.extend(d["KWS"])
      del(d["KWS"])
    else:
      std.append("-")
      std.append("-")
    if "ARG0" in d:
      std.append(d.popitem(False)[1])
    else:
      std.append("-")
    if "ARG1" in d:
      std.append(d.popitem(False)[1])
    else:
      std.append("-")
    for x in range(min(len(d),2)):
      std.append(d.popitem(False)[1])
    if len(std)<9:
      std = std + (["-"]*(9-len(std)))
    stds.append(std)
  return stds
      

def toFloat(x):
  x = x.replace(",","")
  try:
    return float(x)
  except:
    return False

mathKW = ['plus', 'sum', 'difference', 'total', 'change', 'more', 'less', 'additional', 'remain', 'remained', 'fewer', 'remaining', 'together', 'combined', 'increase', 'decrease', 'increased', 'decreased', 'add', 'added', 'product', 'quotient', 'per', 'twice', 'half', 'split into', 'times as many', 'in all', 'have left', 'how many', 'how much']

def get_kws_nums(t,s):
  spl = []
  nums = [x for x in t.split() if toFloat(x)]
  kws = [x for x in mathKW if " "+x+" " in " "+t+" "]
  if not nums and not kws:
    pass
  elif len(s) == 1:
    nums+=["-","-"]
    kws+=["-","-"]
    s[0][1].append(("NUMS",nums[:2]))
    s[0][1].append(("KWS",kws[:2]))
  else:
    ret = []
    ts = t.split(" ")
    idxs = [ts.index(x[0].lower()) for x in s]
    numidx = [ts.index(n) for n in nums]
    kwidx = [len(t.split(" "+k+" ")[0].split(" ")) for k in kws]
    mnum = [max([i for i in idxs if i<n] or [-1]) for n in numidx]
    mkw = [max([i for i in idxs if i<n] or [-1]) for n in kwidx]
    for i,idx in enumerate(idxs):
      thesenums = [x for i,x in enumerate(nums) if mnum[i] == idx]
      thesekw = [x for i,x in enumerate(kws) if mkw[i] == idx]
      thesenums+=["-","-"]
      thesekw+=["-","-"]
      s[i][1].append(("NUMS",thesenums[:2]))
      s[i][1].append(("KWS",thesekw[:2]))

  return s

def doNER(ner,srl,txt):
  txt = txt.split(" ")
  ner = [x.split() for x in ner.split("\n")]
  nertxt = [x[1].lower() for x in ner]
  try:
    assert(len(nertxt)==len(txt))
  except:
    return -1,-1

  last = "O"
  nertxt = []
  for x in ner:
    if x[4] != "O":
      for i in range(len(srl)):
        if x[1] in srl[i]:
          srl[i] = [x[4] if z == x[1] else z for z in srl[i]]

      if x[4] != last:
        nertxt.append(x[4])
    else:
      nertxt.append(x[1].lower())
    last = x[4] 

  return srl," ".join(nertxt)
    

class configMath:
  def __init__(self):
    self.depdir = "data/deps/"
    self.nerdir = "data/splitsNER/"
    self.txtdir = "data/splits/"

def main(cfg):
  bad = 0
  walk = os.walk(cfg.depdir)
  files = next(walk,None)
  k = 0
  srcs = []
  pasts = []
  targs = []
  fns = []
  dps = []
  while files:

    deptexts = [x for x in files[2] if x[-5:] == ".deps"]
    for f in deptexts:
      src = []
      past = []
      targ = []
      dp = []

      fns.append(f)
      fn = files[0]+"/"+f
      with open(cfg.txtdir + f.split(".deps")[0] + ".txt.out") as txt:
        txt_ = txt.read().split('\n')

      with open(cfg.nerdir+f[:-5]+".txt.out.conll") as g:
        ner = g.read().strip().split('\n\n')

      #get srl
      srl = parse(fn)
      i = 0
      last = []
      while i < min(len(srl),len(txt_)):
        if srl[i] != []:

          # get nums and mwks
          t = txt_[i].lower()
          try:
            s = get_kws_nums(t,srl[i])
          except:
            s = srl[i]
          std = standardize(s,t)
          if std == []:
            print(srl[i]);
            print("FAILURE")
            exit()
          asrl,atxt = doNER(ner[i],std,t)
          if asrl == -1:
            print("BAD: ",fn)
            i = max(len(srl),len(txt_))
            bad +=1
            continue
          src.append(asrl)
          past.append(last)
          targ.append(atxt)
          dp.append(std)
          last = asrl
        i+=1
      srcs.append(src)
      pasts.append(past)
      targs.append(targ)
      dps.append(dp)
    files = next(walk,None)
    k +=1

  print(bad)
  with open("pickles/anonMathData-2.23.pickle",'wb') as f:
    pickle.dump((fns,srcs,pasts,targs,dps),f)


if __name__=="__main__":
  cfg = configMath()
  main(cfg)
