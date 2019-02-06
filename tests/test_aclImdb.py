def test_preprocessed_aclImdb():
  from srmnlp.data.aclImdb import preprocessed_aclImdb
  train_batches, test_batches = preprocessed_aclImdb(batch_size=16)

  assert isinstance(train_batches, type([]))
  assert isinstance(test_batches, type([]))
  assert len(train_batches)
  assert len(test_batches)
  assert len(train_batches[-1][0]) == 16
  assert len(test_batches[-1][0]) == 16
