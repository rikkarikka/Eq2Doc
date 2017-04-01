import sys

# usage python3 stitch_together.py <sents> <align>

try:
  sentsf = sys.argv[1]
  alignf = sys.argv[2]
  
except:
  print("usage python3 stitch_together.py <sents> <align>")

def main():
  with open(sentsf) as f:
    sents = f.readlines()
  with open(alignf) as f:
    align = [x.strip() for x in f.readlines()]

  last = align[0]
  doc = sents[0].strip() + " "
  for i in range(1,len(align)):
    l = align[i]
    if last!=l:
      print(doc[:-1])
      doc = ""
      last = l
    doc += sents[i].strip()+" "
  print(doc[:-1])
  
main()

