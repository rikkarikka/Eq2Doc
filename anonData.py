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

  srl = []
  for i,s in enumerate(sents):
    t_srl = {}
    if not s: 
      srl.append([])
      continue
    trips = [x.split()[0:5] for x in s.split("\n")]
    preds = set([" ".join(x[0:2]) for x in trips if "ARG" in x[2]])
    for p in preds:
      ps = p.split(" ")
      args = [x for x in trips if x[0] == ps[0] and x[1] == ps[1] and "ARG" in x[2]]
      for a in args:
        appnd = tuple(a[2:4])
        if p not in t_srl:
          t_srl[p] = []
        t_srl[p].append(appnd)
      rargs = [x for x in trips if x[3] == ps[0] and x[4] == ps[1]]
      for a in rargs:
        appnd = (a[2],a[0])
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

def standardize(srl):
  stds = []
  for p,l in srl:
    std = [p]
    l.sort()
    d = dict(l)
    #deal with arg3/4
    ar = "ARG3"
    for k in ["ARG3","ARG4","LOC","TMP"]:
      if k in d:
        d[ar] = d[k]
        if ar == "ARG4": break
      ar = "ARG4"

    #deal w/ mathkws
    try:
      kws = d["NUMS"] + d["KWS"]
      kws = [x for x in kws if x!='-']
    except:
      pass

    for i in range(4):
      if "ARG"+str(i) in d:
        std.append(d["ARG"+str(i)])
      else:
        std.append("-")

    if "DIS" in d:
      std.append(d["DIS"])
    else:
      std.append("-")

    if "NEG" in d:
      std.append(d["NEG"])
    else:
      std.append("-")

    for i in range(2):
      try:
        std.append(kws[i].replace(" ","_"))
      except:
        std.append('-')

    assert(len(std)==9)
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
  t = t.lower()
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

def nertext(ner,nerd,txt):
  txt = " ".join(txt)
  txt2 = [x.split(" ") for x in txt]
  ner = "\n".join(ner)
  ner = [x.split() for x in ner.split("\n")]
  d = nerd
  nertxt = [x[1].lower() for x in ner]
  last = "O"
  nertxt = []
  thisner = []
  for x in ner:
    if x[4] != "O":
      nah = True
      if x[4] in d:
        if x[1] in d[x[4]]:
          replacement = d[x[4]].index(x[1])
          rstring = x[4]+"_"+str(replacement)
          nah = False
          
      if nah:
        if x[4]!=last:
          rstring = x[4]
    else:
      if last != "O":
        nertxt.append(rstring)
      nertxt.append(x[1].lower())
    last = x[4] 

  return " ".join(nertxt)


def nersrl(ner,srl,nerd):
  ner = [x.split() for x in ner.split("\n")]
  last = "O"
  tag = None
  d = nerd if nerd else {}
  for x in ner:
    if x[4] != "O":
      if x[4] not in d:
        d[x[4]] = []
      rstring = x[4]
      for i in range(len(srl)):
        if x[1] in srl[i]:
          if x[1] in d[x[4]]:
            replacement = d[x[4]].index(x[1])
          else:
            d[x[4]].append(x[1])
            replacement = len(d[x[4]])-1
          rstring = x[4]+"_"+str(replacement)
          srl[i] = [rstring if z == x[1] else z for z in srl[i]]

    last = x[4] 

  for i in range(len(srl)):
    for j in range(len(srl[i])):
      w = srl[i][j] if "_" not in srl[i][j] else srl[i][j].split("_")[0]
      if w not in d:
        srl[i][j] = srl[i][j].lower()
  return srl,d 
    
def srl_lower(srl):
  for s in range(len(srl)):
    for p in range(len(srl[s])):
      srl[s][p] = [x.lower() for x in srl[s][p]]
  return srl


def main(df,pn):
  bad = 0
  walk = os.walk(df+'/txtdep')
  files = next(walk,None)
  k = 0
  srcs = []
  targs = []
  fns = []
  while files:
    plaintexts = [x for x in files[2] if x[-5:] == ".deps"]
    for f in plaintexts:
      src = []
      targ = []

      fns.append(f)
      fn = files[0]+"/"+f
      with open(fn.split(".deps")[0]) as txt:
        txt_ = txt.read().split('\n')

      with open(df+"/ner/"+f[:-5]+".conll") as g:
        #print(f)
        ner = g.read().strip().split('\n\n')

      #get srl
      srl = parse(fn)

      #build ner dict
      i = 0
      nerd = None
      while i < min(len(srl),len(txt_)):
        if srl[i] != []:

          # get nums and mwks
          t = txt_[i].lower()
          try:
            s = get_kws_nums(t,srl[i])
          except:
            s = srl[i]
          std = standardize(s)
          if std == []:
            print(srl[i]);
            print("FAILURE")
            exit()
          asrl,nerd = nersrl(ner[i],std,nerd)
          src.append(asrl)
        i+=1
      try:
        targ = nertext(ner,nerd,txt_)
        srcs.append(src)
        targs.append(targ)
      except:
        bad+=1
    files = next(walk,None)
    k +=1
    #if k > 100: break

  print("Failed: ",bad)
  with open("pickles/"+pn+".pickle",'wb') as f:
    pickle.dump((fns,srcs,targs),f)


if __name__=="__main__":
  datafolder = sys.argv[1]
  try:
    picklename = sys.argv[2]
  except:
    picklename = "data"
  main(datafolder,picklename)
