import torch


def test_create_vocabulary():
  from srmnlp.data.preprocess import create_vocabulary
  vocab = create_vocabulary([
      (['a', 'b', 'c'], '+'),
      (['d', 'e', 'f'], '+'),
      (['a', 'b', 'k'], '+'),
      (['a', 'i', 'k'], '+'),
      (['e', 'j', 'k'], '+'),
      (['a', 'b', 'k'], '+'),
      ]
      , 3)

  assert set(vocab) == set(['UNK', 'PAD', 'a', 'b', 'k'])


def test_make_batches():
  from srmnlp.data.preprocess import make_batches
  tl = []
  for i in range(20):
    tl.append((list(range(i)), '+'))
  batches = make_batches(tl, 4)
  assert isinstance(batches, type([]))
  assert len(batches) == 5
  assert isinstance(batches[-1], type((6, '.', 9)))
  assert isinstance(batches[-1][0], type(torch.tensor(6.9)))
  assert batches[-1][0].size() == torch.Size([4, 19])


def test_pad():
  from srmnlp.data.preprocess import pad
  t, labels = pad([
    ([0], '+'),
    ([1, 2], '-'),
    ([1, 2, 3], '+')
    ])
  assert isinstance(t, type(torch.tensor(6.9)))
  assert t.size() == torch.Size([3, 3])


def test_index():
  from srmnlp.data.preprocess import index
  indexed = index('a k a b', { 'a' : 2, 'b' : 3, 'UNK' : 1, 'PAD' : 0 })
  assert indexed == [ 2, 1, 2, 3 ]
  indexed = index(['a', 'k', 'a', 'b'],
      { 'a' : 2, 'b' : 3, 'UNK' : 1, 'PAD' : 0 }
      )
  assert indexed == [ 2, 1, 2, 3 ]


def test_word_to_index():
  from srmnlp.data.preprocess import word_to_index
  w2i = { 'a' : 2, 'b' : 3, 'UNK' : 1, 'PAD' : 0 }
  assert word_to_index('c', w2i) == 1
  assert word_to_index('b', w2i) == 3


def test_tokenzie():
  from srmnlp.data.preprocess import tokenize
  assert tokenize('a b c') == ['a', 'b', 'c']


def test_preprocess():
  from srmnlp.data.aclImdb import truncated_data
  from srmnlp.data.preprocess import create_vocabulary
  from srmnlp.data.preprocess import make_batches
  from srmnlp.data.preprocess import index
  from srmnlp.data.preprocess import vocab_to_w2i

  train, test = truncated_data()
  samples = test
  vocab = create_vocabulary(samples, 100)
  w2i = vocab_to_w2i(vocab)
  indexed_samples = [ (index(text, w2i), label) for text, label in samples ]
  batches = make_batches(indexed_samples, 10)
  assert len(batches) == 10
  assert isinstance(batches[-1], type((6, '.', 9)))
  assert isinstance(batches[-1][0], type(torch.tensor([6.9])))
