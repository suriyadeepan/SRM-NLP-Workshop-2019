import torch
import torch.nn.functional as F


def test_LstmClassifier():
  from srmnlp.reduce.lstm import LstmClassifier
  hparams = {
    'vocab_size'  : 100,
    'emb_dim'     : 100,
    'hidden_dim'  : 32,
    'lr'          : 2e-5,
    'output_size' : 2,
    'loss_fn'     : F.cross_entropy,
    'batch_size'  : 8
    }

  lstm_out = LstmClassifier(hparams)(torch.randint(0, 100, (8, 11)))

  assert isinstance(lstm_out, type(torch.tensor([6.9])))
  assert lstm_out.size() == torch.Size([8, 2])
  assert int(F.softmax(lstm_out, dim=-1).sum().item()) == 8.  # almost 7.9999 happens
