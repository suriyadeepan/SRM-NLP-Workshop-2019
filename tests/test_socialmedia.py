def test_socialmedia():
  from srmnlp.data.socialmedia import socialmedia
  train, test = socialmedia()
  assert isinstance(train, type([]))
  assert isinstance(test, type([]))
  assert isinstance(train[-1], type((6, 9)))
  assert isinstance(test[-1], type((6, 9)))
  assert len(train) == 5000
  assert len(test) == 7086 - 5000


def test_preprocessed_aclImdb():
  from srmnlp.data.socialmedia import preprocessed_socialmedia
  (train_batches, test_batches), vocab, w2i = preprocessed_socialmedia(batch_size=16)

  assert len(vocab)
  assert isinstance(train_batches, type([]))
  assert isinstance(test_batches, type([]))
  assert len(train_batches)
  assert len(test_batches)
  assert len(train_batches[-1][0]) == 16
  assert len(test_batches[-1][0]) == 16
