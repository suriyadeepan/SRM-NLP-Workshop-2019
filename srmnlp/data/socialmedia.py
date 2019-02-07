from random import shuffle


def cleanup(text):
  return text.replace('\n', '')


def socialmedia(filename='data/socialmedia/training.txt', split_at=5000):
    lines = open(filename).readlines()
    samples = [ tuple(reversed(cleanup(line).split('\t'))) for line in lines ]
    shuffle(samples)
    return samples[:split_at], samples[split_at:]


def preprocessed_socialmedia(batch_size=32, truncate=True,
  label2idx={ '1' : 1, '0' : 0 }):

  from srmnlp.data.preprocess import create_vocabulary
  from srmnlp.data.preprocess import make_batches
  from srmnlp.data.preprocess import index
  from srmnlp.data.preprocess import vocab_to_w2i

  train, test = socialmedia()
  train = [ (x, label2idx[y]) for x, y in train ]
  test = [ (x, label2idx[y]) for x, y in test ]
  vocab = create_vocabulary(train, vocab_max_size=800)
  w2i = vocab_to_w2i(vocab)
  indexed_train = [ (index(text, w2i), label) for text, label in train ]
  indexed_test = [ (index(text, w2i), label) for text, label in test ]
  return (
      make_batches(indexed_train, batch_size),
      make_batches(indexed_test, batch_size)
      ), vocab, w2i


if __name__ == '__main__':
  (train, test), vocab, w2i = preprocessed_socialmedia(batch_size=3)
  text, label = train[0]
  print('Text  :', text.size(), text)
  print('Label :', label.size(), label)
