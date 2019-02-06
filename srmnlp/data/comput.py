from random import shuffle


def comput(filename='data/comput.txt'):
  text = open(filename).read().replace('\n', ' ')
  words = text.split()
  samples = []
  for i in range(2, len(words) - 2):
    context = [ words[i - 2], words[i - 1], words[i + 1], words[i + 2] ]
    target = words[i]
    samples.append((context, target))

  # return vocab_to_w2i(vocab)
  shuffle(samples)

  return samples, None


def preprocessed_comput(batch_size=1):
  from srmnlp.data.preprocess import create_vocabulary
  from srmnlp.data.preprocess import make_batches
  from srmnlp.data.preprocess import index
  from srmnlp.data.preprocess import vocab_to_w2i

  samples, _ = comput()
  vocab = create_vocabulary(samples)[2:]  # we do not need PAD and UNK here
  w2i = vocab_to_w2i(vocab)
  label2idx = w2i
  samples = [ (x, label2idx[y]) for x, y in samples ]
  indexed_samples = [ (index(text, w2i), label) for text, label in samples ]
  return (
      make_batches(indexed_samples, batch_size),
      None
      ), vocab, w2i


if __name__ == '__main__':

  print(preprocessed_comput())
