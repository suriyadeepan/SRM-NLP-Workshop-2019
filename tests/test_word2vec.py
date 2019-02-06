import torch


def test_CBOW():
  from srmnlp.repr.word2vec import CBOW
  cbow = CBOW(100, 10)
  assert cbow(torch.randint(0, 100, [1, 4])).sum()
  assert cbow.get_embedding(96).sum()
