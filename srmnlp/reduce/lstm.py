import torch
import torch.nn as nn

from torch.autograd import Variable


class LstmClassifier(nn.Module):

  def __init__(self, hparams, weights=None):
    """
    LSTM RNN Classifier

    Args:
      hparams : dictionary of hyperparameters

    """
    super(LstmClassifier, self).__init__()

    self.hparams = hparams
    self.weights = weights
    # init embedding lookup
    self.embedding = nn.Embedding(hparams['vocab_size'], hparams['emb_dim'])
    # set learned weights
    #  disable training
    if weights:
      self.embedding.weight = nn.Parameter(weights['glove'], requires_grad=False)
    # lstm
    self.lstm = nn.LSTM(hparams['emb_dim'], hparams['hidden_dim'])
    # linear layer
    self.linear = nn.Linear(hparams['hidden_dim'], hparams['output_size'])

  def forward(self, sequence, batch_size=None, get_hidden=False):
    """
    Forward Operation.

    Args:
      sequence : list of indices based off a sentence

    """
    # infer batch_size and seqlen
    #  print(sequence.size())
    # restructure sequence
    #  sequence = sequence.permute(1, 0)
    # embed input
    input = self.embedding(sequence)
    input = input.permute(1, 0, 2)

    # initial state
    batch_size = batch_size if batch_size else self.hparams['batch_size']
    if torch.cuda.is_available():
      h0 = Variable(torch.zeros(1, batch_size, self.hparams['hidden_dim']).cuda())
      c0 = Variable(torch.zeros(1, batch_size, self.hparams['hidden_dim']).cuda())
    else:
      h0 = Variable(torch.zeros(1, batch_size, self.hparams['hidden_dim']))
      c0 = Variable(torch.zeros(1, batch_size, self.hparams['hidden_dim']))

    # fix for "RNN weights not part of single contiguous chunk of memory" issue
    #  https://discuss.pytorch.org/t/rnn-module-weights-are-not-part-of-single-contiguous-chunk-of-memory/6011/13
    self.lstm.flatten_parameters()
    lstm_out, (h, c) = self.lstm(input, (h0, c0))

    # expose final state/representation
    self.h = h[-1]

    # linear layer
    linear_out = self.linear(h[-1])

    # softmax layer
    # softmax_out = F.log_softmax(linear_out, dim=-1)

    if get_hidden:
      return linear_out, self.h

    return linear_out
