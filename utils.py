mathKW = ['plus', 'sum', 'difference', 'total', 'change', 'more', 'less', 'additional', 'remain', 'remained', 'fewer', 'remaining', 'together', 'combined', 'increase', 'decrease', 'increased', 'decreased', 'add', 'added', 'product', 'quotient', 'per', 'twice', 'half', 'split into', 'times as many', 'in all', 'have left', 'how many', 'how much']

def toFloat(x):
  x = x.replace(",","")
  try:
    return float(x)
  except:
    return False
