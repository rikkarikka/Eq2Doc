import torch
import opts

class net(nn.Module):
  def __init__(self):
    self.emb = nn.Embeddings(opts.vocab_size, opts.h_size)
    self.brnn = nn.LSTM(opts.h_size,opts.h_size,batch_first=True,bidirectional=True)
    self.bsz = opts.bsz

  def init_hidden(self,bsz=None):
    if not bsz:
      bsz = self.bsz
    if CUDA_ON:
      h = Variable(torch.cuda.FloatTensor(2,bsz,opts.h_size).zero_(),requires_grad=False)
      c = Variable(torch.cuda.FloatTensor(2,bsz,opts.h_size).zero_(),requires_grad=False)
    else:
      h = Variable(torch.Tensor(2,bsz,opts.h_size).zero_(),requires_grad=False)
      c = Variable(torch.Tensor(2,bsz,opts.h_size).zero_(),requires_grad=False)
    return h,c

  def forward(self,inp,hc):
    inp_emb = self.emb(inp)
    o,(h,c) = self.brnn(inp_emb,hc)
    return h,c
    

def load_data():
  train_h, train_c = torch.load("train-encodings.pt")
  val_h, val_c = torch.load("val-encodings.pt")
  with open(opts.eq_train_data) as f:
    train_eq = f.read().split("\n")
  with open(opts.eq_val_data) as f:
    val_eq = f.read().split("\n")

  assert(len(train_eq)==train_h.size(1))
  assert(len(val_eq)==val_h.size(1))

  return (train_h,train_c),(dev_h,dev_c),train_eq,val_eq

  


def main():
  m = net()
  train_tgts,val_tgts,train_src,val_src = load_data()

if __name__=="__main__":
  main()
  
