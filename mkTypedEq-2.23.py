import sys
import os
import pickle
import json
import re
from collections import Counter, OrderedDict, defaultdict
from utils import *

with open("data/math_data/MathProblems.json") as f:
  jdata = json.load(f)

copulas = ['is','am','are','was','were','be','being','been']
breakwords = " when if , and or since but . ? ".split()

def toFloat(x):
  x = x.replace(",","")
  try:
    return float(x)
  except:
    return False

def float_txt(t,eq):
  tl = t.split("\n")
  ret = []
  for l in tl:
    ret.append(' '.join([str(toFloat(w)) if (toFloat(w) and str(toFloat(w)) in eq) else w for w in l.split(" ")]))
  return "\n".join(ret).lower()


def clean_eq(eq):
  for c in "+=-*()/":
    eq = eq.replace(c," "+c+" ")
  eq = eq.replace("  "," ")
  e = eq.split()
  varis = [x for x in e if not x[0].isdigit() and x not in "+=-*()/;"]
  for j,v in enumerate(varis):
    eq = eq.replace(" "+v+" "," "+chr(ord("X")+j)+" ")
    if eq[:len(v)]==v:
      eq = chr(ord("X")+j)+" "+eq[len(v):]
    if eq[-len(v):]==v:
      eq = eq[:-len(v)]+chr(ord("X")+j)+" "
  e = eq.lower().split()
  e = [str(toFloat(x)) if toFloat(x) else x for x in e]
  return e

def fallback(sidx,pidx):
  with open("data/math_data/parses/"+str(pidx)+".txt.json") as f:
    parse = json.load(f)
  s_parse = parse["sentences"][sidx]
  root_ = [x for x in s_parse["collapsed-ccprocessed-dependencies"] if x['governorGloss'] == "ROOT"][0]
  rootword = root_['dependentGloss']
  childs = [x for x in s_parse["collapsed-ccprocessed-dependencies"] if x['governor'] == root_['dependent']]
  d = {rootword:[]}
  for c in childs:
    print("FALLBACK",c)
    if c['dep']=='punct':continue
    if 'nmod' in c['dep'] or "case" in c['dep']:
      arg = "TMP/LOC"
    elif 'nsubj' in c['dep']:
      arg = 'ARG0'
    elif 'dobj' in c['dep']:
      arg = "ARG1"
    else:
      arg = "ARG2"
    d[rootword].append([arg, c['dependentGloss']])
  return d

def process_parse(parse,txt,pidx):
  sents = parse.strip().split("\n\n")
  whole_parse = []
  srl = []
  for i,s in enumerate(sents):
    t_srl = {}
    if s.strip() == "NO DEPS":
      srl.append(fallback(i,pidx))
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
            appnd = aftr[0]
          else:
            appnd = [a[2],a[0]]
        elif "ARG" not in a[2] and a[2] != "NONE":
          append = [a[2],a[0]]
        else:
          appnd = a[2:4]
        
        if p not in t_srl:
          t_srl[p] = []
        t_srl[p].append(appnd)

    # deal w/ copulas
    '''
    cops = [x[0:2] for x in trips if x[0] in copulas]
    print(t_srl)
    for c in cops:
      print(c)
      deps = [x[3]+" "+x[4] for x in trips if x[0:2] == c]
      if any([x in t_srl for x in deps]):
        print("continuing")
        continue
      else:
        l = sorted([x for x in trips if x[0:2] == c],key=lambda x : x[4])
        print(l)
        t_srl[" ".join(c)] = [x[2:4] for x in l]
    '''
    if t_srl == {}:
      t_srl = fallback(i,pidx)
    assert(t_srl!={})
    srl.append(t_srl)
  print(len(srl),len(txt.strip().split('\n')))
  assert(len(srl)==len(txt.strip().split('\n')))
    
  return whole_parse,srl

def type_eq(eq,parse):
  print(eq)
  types = []
  for s in parse:
    nums = sorted([x for x in s if str(toFloat(x[0])) in eq and not x[3][0].isdigit()],key= lambda x:int(x[4]),reverse=True)
    for n in nums:
      num = str(toFloat(n[0]))
      if num in eq:
        eq[eq.index(num)] = num+" "+n[3]
        types.append(n[3])

    dollar_nums = [x for x in s if x[0]=="$" and str(toFloat(x[3])) in eq]
    for n in dollar_nums:
      num = str(toFloat(n[3]))
      eq[eq.index(num)] = "$ "+num
      types.append("$")

  # do x

  ent = False
  many = [x[3] for x in parse[-1] if x[0] == "many"]
  known_many = [x for x in many if x in types]
  if known_many:
    ent = known_many[0]
  elif many:
    ent = many[0]
  else:
    if len(set(types)) == 1: 
      if types[0] == "$":
        ent = "$"
      else:
        ent = types[0]

    if not ent:
      much = sorted([x for x in parse[-1] if x[3] == "much"],key=lambda x:int(x[1]),reverse=True)
      known_much = [x[0] for x in much if x[0] in types]
      if known_much:
        ent = known_much[0]
      elif much:
        if "$" in types or "dollars" in types:
          ent = "$"
        else:
          ent = much[0][0]

    if not ent:
      if "$" in types:
        ent = "$"

  if ent == "$":
    eq[eq.index("x")] = "$ x"
  elif ent:
    eq[eq.index("x")] = "x "+ent

  return eq

def make_plan(srl,txt,eq):
  eq = ' '.join(eq).split()
  txt = txt.strip().split("\n")
  for i,ps in enumerate(srl):
    s = enumerate(txt[i].split())
    mwrds = [x for x in s if x[1] in eq]
    kws = [x for x in mathKW if x in txt[i].lower()] 
    if len(ps) == 1:
      mwrds = [x[1] for x in mwrds]
      key = list(ps.keys())[0]
      ps[key].append(['kws',mwrds+kws])
      # not in same sentence as previous pred
      ps[key].append(['sent','0'])
    else:
      psort = sorted([(int(x[0].split(" ")[1]),x) for x in ps.items()], reverse = True)
      for idx,x in psort:
        if (idx,x) == psort[-1]:
          x[1].append(['sent','0'])
        else:
          x[1].append(['sent','1'])
        xmwrds = [y for y in mwrds if y[0]>idx]
        xkws = [y for y in kws if len(txt[i].split(y)[0].split(" "))>idx]
        x[1].append(['kws',[y[1] for y in xmwrds]+xkws])
        mwrds = [y for y in mwrds if y not in xmwrds]
        kws = [y for y in kws if y not in xkws]
  # make plan string
  preds = []
  assert(len(srl)==len(txt))
  breaks = []
  for ps in srl:
    items = [x for x in ps.items() if ['sent','0'] in x[1]]
    breaks.extend(items)
  assert(len(breaks)==len(srl))

  for ps in srl:
    if len(ps)==1:
      key = list(ps.keys())[0]
      preds.append((key,ps[key]))
    else:
      psort = sorted([(int(x[0].split(" ")[1]),x) for x in ps.items()])
      for i,x in psort:
        preds.append((x[0],x[1]))

  plan = []
  print(preds)
  for p,vals in preds:
    pstring = [p.split(" ")[0]]
    if "ARG0" in [x[0] for x in vals]:
      pstring.extend([x[1] for x in vals if x[0] == "ARG0"])
    else:
      pstring.append("NONE")
    otherargs = sorted([x for x in vals if "ARG" in x[0] and "0" not in x[0]])
    others = sorted([x for x in vals if all([y not in x[0] for y in "ARG sent kws".split()])],reverse=True)
    pstring.extend([x[1] for x in otherargs])
    pstring.extend([x[1] for x in others])
    '''
    nones = [x[1] for x in vals if "ARG" not in x[0] and x[0] not in ["TMP","LOC",'sent','kws']]
    pstring.extend(nones)
    '''
    if len(pstring)<args+1:
      pstring.extend(["NONE"]*((args+1)-len(pstring)))
    pstring = pstring[:args+1]
    assert(len(pstring)==args+1)
    '''
    kws = [x[1] for x in vals if x[0] == 'kws'][0]
    kws = sorted([x for x in kws if x not in pstring])
    needed = [x for x in kws if toFloat(x)]
    if needed: 
      kws = needed
      pstring.extend([x for x in needed])
    kws = [x for x in kws if x not in needed]
    for w in kws:
      if " " in w:
        if "split" in w or "times" in w:
          w = w.split()[0]
        else:
          w = w.split()[1]
      pstring.append(w)
    
    if len(pstring)<args+kwlen+1:
      pstring.extend(["NONE"]*((args+kwlen+1)-len(pstring)))
    pstring = pstring[:args+kwlen+1]
    assert(len(pstring)==args+kwlen+1)
    '''
    sent = [x for x in vals if x[0] == 'sent']
    pstring.append(sent[0][1])
    
    plan.append(pstring)
  breaks = [x for x in plan if x[-1]=="0"]
  assert(len(breaks)==len(txt))
  plan = [[x.lower() if x != "NONE" else x for x in y] for y in plan]
  if len(plan)>6:
    undeletable = [x for x in plan if (x[-1]=="0") or (not all([y=="NONE" for y in x[-(kwlen+1):-1]]))]
    if len(undeletable)>6:
      undeletable = [x for x in plan if not all([y=="NONE" for y in x[-(kwlen+1):-1]])]
    plan = undeletable
  if len(plan)>6:
    print(plan)
    input()
  return plan




def get_eq(fn):
  fn = int(fn.split(".")[0])
  jdat = [x for x in jdata if x['iIndex']==fn]
  if not jdat:
    return -1
  eq = jdat[0]['lEquations'][0]
  eq = clean_eq(eq)
  print(fn)
  with open("data/math_data/splits/"+str(fn)+".txt.out") as f:
    txt = float_txt(f.read(),eq)
  print(txt)
  with open("data/math_data/deps/"+str(fn)+".deps") as f:
    parse,srl = process_parse(f.read(),txt,fn)
  

  if parse:
    eq = type_eq(eq,parse)

  eq = " ".join(eq).split(" ")

  # order numbers by apperaance in txt
  eqnums = [toFloat(x) for x in eq if toFloat(x)]
  txtnums = [toFloat(x) for x in jdat[0]['sQuestion'].split(" ") if toFloat(x) in eqnums]
  eq = ["NUMBER_"+str(txtnums.index(toFloat(x))) if toFloat(x) in txtnums else x for x in eq]

  return eq

def main():
  with open("pickles/math.pickle",'rb') as f:
    fns,srcs,targs = pickle.load(f)

  herefns = []
  eqs = []
  heredps = []
  hereTargets = []
  for i in range(len(fns)):
    f = fns[i]
    dp = srcs[i]
    eq = get_eq(f)
    print(eq)
    if eq == -1:
      continue
    else:
      herefns.append(f)
      heredps.append(dp)
      eqs.append(eq)
      hereTargets.append(targs[i])

  with open("pickles/eq2planData.pickle",'wb') as f:
    pickle.dump((herefns,eqs,heredps,hereTargets),f)

if __name__=="__main__":
  main()
