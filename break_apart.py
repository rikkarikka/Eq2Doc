import sys

def usage():
  print("python3 break_apart.py <src_doc> \n output is src_doc_sents and src_sent_map ")

def breakup(docs):
  m = []
  sents = []
  for i,d in enumerate(docs):
    tmp = d.strip().split(" EOS ")
    sent.extend(tmp)
    m.extend([i]*len(tmp))
  return sents, m

if __name__=="__main__":
  try:
    src_doc = sys.argv[1]
  except:
    usage();exit()

  with open(src_doc) as f:
    docs = f.readlines()
  sents,m = breakup(docs)

  with open(src_doc+"_sents.txt") as f:
    f.write("\n".join(sents))

  with open(src_doc+"_sents_map.txt") as f:
    f.write("\n".join(m))
