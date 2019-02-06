import torch
import torch.nn as nn

import torch.nn.functional as F


class CBOW(torch.nn.Module):

  def __init__(self, vocab_size, embed_dim):
    super(CBOW, self).__init__()

    # create embeddings with random initialization
    self.embeddings = nn.Embedding(vocab_size, embed_dim)
    # linear layer (1)
    self.linear1 = nn.Linear(embed_dim, 128)
    # linear layer (2)
    self.linear2 = nn.Linear(128, vocab_size)

  def forward(self, inputs):
    embedded = self.embeddings(inputs).sum(dim=1)
    out = F.relu(self.linear1(embedded))
    return F.log_softmax(self.linear2(out), dim=-1)

  def get_embedding(self, idx):
    if not isinstance(type(idx), type(torch.tensor(6.9))):
      idx = torch.tensor(idx)
    return self.embeddings(idx)
