import os
import nltk
from tqdm import tqdm
from random import shuffle
import pickle


TOKENIZER = nltk.word_tokenize


def cleanup(text):
  return text.replace('<br />', '')


def make_sample(filename, label):
  return ( TOKENIZER(
    cleanup(
      open(filename).read()
      )
    ), label )


def aclImdb(sample_from_train=False):
  imdb_senti_data_path = 'data/aclImdb/{}/{}/'
  train, test = [], []

  dtypes = ['train', 'test']
  if sample_from_train:
    dtypes = ['train']

  for dtype in dtypes:
    for label in ['pos', 'neg']:
      path = imdb_senti_data_path.format(dtype, label)
      for filename in tqdm(os.listdir(path)[:1000]):
        sample = make_sample(os.path.join(path, filename), label)
        if dtype == 'train':
          train.append(sample)
        else:
          test.append(sample)

  return train, test


def truncate_aclImdb(train_samples=1000, test_samples=500):
  train, _ = aclImdb(sample_from_train=True)
  samples = sorted(train, key=lambda x : len(x[0]))
  shuffle(samples)
  return ( samples[:train_samples],
      samples[train_samples:train_samples + test_samples]
      )


def load_samples(filename):
  with open(filename, 'rb') as f:
    return pickle.load(f)


def truncated_data(filename='data/aclImdb_truncated.{}'):
  return ( load_samples(filename.format('train')),
      load_samples(filename.format('test'))
      )


def dump_dataset(dataset, name='aclImdb_truncated', path='data/'):
  train, test = dataset
  print(test)
  dump_samples(train, os.path.join(path, name + '.train'))
  dump_samples(test , os.path.join(path, name + '.test' ))


def dump_samples(samples, filename):
  with open(filename, 'wb') as f:
    pickle.dump(samples, f)


def preprocessed_aclImdb(batch_size=32, truncate=True, label2idx={ 'pos' : 1, 'neg' : 0 }):
  from srmnlp.data.preprocess import create_vocabulary
  from srmnlp.data.preprocess import make_batches
  from srmnlp.data.preprocess import index
  from srmnlp.data.preprocess import vocab_to_w2i

  train, test = truncated_data() if truncate else aclImdb()
  train = [ (x, label2idx[y]) for x, y in train ]
  test = [ (x, label2idx[y]) for x, y in test ]
  vocab = create_vocabulary(train)
  w2i = vocab_to_w2i(vocab)
  indexed_train = [ (index(text, w2i), label) for text, label in train ]
  indexed_test = [ (index(text, w2i), label) for text, label in test ]
  return (
      make_batches(indexed_train, batch_size),
      make_batches(indexed_test, batch_size)
      ), vocab, w2i
