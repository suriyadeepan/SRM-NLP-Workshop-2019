import nltk
import torch


TOKENIZER = nltk.word_tokenize


def tokenize(sentence, tokenizer=TOKENIZER):
  return TOKENIZER(sentence)


def word_to_index(w, w2i, UNK_IDX=1):
  return w2i[w] if w in w2i else UNK_IDX


def index(sentence, w2i):
  # is it a string ?
  if isinstance(sentence, type('cosimo')):
    sentence = tokenize(sentence)
  # it should be a list now
  assert isinstance(sentence, type([]))

  return [ word_to_index(token, w2i) for token in sentence ]


def pad(samples, max_len=None, PAD_IDX=0, to_tensor=torch.tensor):
  max_len = max([len(s[0]) for s in samples]) if not max_len else max_len
  padded_text = []
  labels = []
  for text, label in samples:
    padded_text.append(
        text + [PAD_IDX] * (max_len - len(text))
       )
    labels.append(label)
  return (
      to_tensor(padded_text) if to_tensor else padded_text,
      to_tensor(labels)
      )


def make_batches(samples, batch_size):
  batches = []
  num_batches = len(samples) // batch_size
  for i in range(num_batches):
    batches.append(
        pad(samples[i * batch_size : (i + 1) * batch_size])
        )
  return batches


def create_vocabulary(samples, vocab_max_size=2000):
  words = []
  for (text, label) in samples:
    words.extend(text)

  fdist = nltk.FreqDist(words)

  return ['PAD', 'UNK'] + \
      [ w for w, f in fdist.most_common(vocab_max_size) ]


def vocab_to_w2i(vocab):
  return { w: i for i, w in enumerate(vocab) }
