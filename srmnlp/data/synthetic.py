import random
import numpy as np


vocab = [
    'zero',   # 0
    'one',    # 1
    'two',    # 2
    'three',  # 3
    'four',   # 4
    'five',   # 5
    'six',    # 6
    'seven',  # 7
    'eight',  # 8
    'nine'    # 9
    ]


def numbers(num_lines=10, min_sent_len=2, max_sent_len=10):
  corpus = []
  for line in range(num_lines):
    sent_len = random.randint(min_sent_len, max_sent_len)
    indices = np.random.randint(0, len(vocab), [sent_len, ])
    corpus.append(' '.join([ vocab[i] for i in indices ]))
  return corpus
