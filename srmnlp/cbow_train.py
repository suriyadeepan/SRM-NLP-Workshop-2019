import torch
import torch.nn as nn

from srmnlp.repr.word2vec import CBOW
from srmnlp.data.comput import preprocessed_comput


def train(model, trainset, num_epochs=50):

  loss_fn = nn.NLLLoss()
  optim = torch.optim.Adam([ p for p in model.parameters() if p.requires_grad ])
  for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.
    for context, target in trainset:
      optim.zero_grad()
      logloss = model(context)
      loss = loss_fn(logloss, target)
      loss.backward()
      optim.step()
      epoch_loss += loss.item()

    print(epoch, epoch_loss)


if __name__ == '__main__':
  # get data
  (trainset, _), vocab, w2i = preprocessed_comput()
  # create model
  cbow = CBOW(len(vocab), 100)
  # train model
  train(cbow, trainset)

  # create context vector
  from srmnlp.data.preprocess import index
  context_words = ['People', 'create', 'to', 'direct']
  context = torch.tensor(index(context_words, w2i)).view(1, -1)
  # get argmax
  _, tidx = torch.max(cbow(context), 1)
  print('Context Words : ', context_words)
  print('Target Word : ', vocab[int(tidx.item())])
