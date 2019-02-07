import torch
import torch.nn.functional as F

from torch.autograd import Variable

import argparse
import os

from srmnlp.reduce.lstm import LstmClassifier
from srmnlp.data.aclImdb import preprocessed_aclImdb
from srmnlp.data.socialmedia import preprocessed_socialmedia

MODEL_SAVE_FILE = 'model.bin'


# cmd line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--mode", default='train', help='run mode : train/evaluate/predict')
parser.add_argument("--data", default='socialmedia', help='choose a dataset to fit on')
parser.add_argument("--embed_dim", default=200, help='embedding dimensions')
parser.add_argument("--hidden_dim", default=256, help='size of hidden state')
parser.add_argument("--batch_size", default=32, help='batch size for training')
parser.add_argument("--input",
    # default='Ohh, such a ridiculous movie. Not gonna recommend it to anyone. Complete waste of time and money.',
    default='This is one of the best creation of Nolan. I can say, it\'s his magnum opus. Loved the soundtrack and especially those creative dialogues.',
    help='input sentence to run prediction on'
    )
args, unknown = parser.parse_known_args()


def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)


def train_epoch(model, train_iter, hparams):
  # prepare model for training
  if torch.cuda.is_available():
    model.cuda()
  # train mode
  model.train()

  # loss function
  loss_fn = hparams['loss_fn']

  optim = torch.optim.Adam([ p for p in model.parameters() if p.requires_grad ])
  steps = 0
  epoch_loss, epoch_accuracy = 0, 0
  for idx, batch in enumerate(train_iter):
    # (1) clear gradients
    optim.zero_grad()  # NOTE : why did I do model.zero_grad() ?

    # (2) inputs and targets
    inputs, targets = batch
    # if cuda
    if torch.cuda.is_available():
      inputs = inputs.cuda()
      targets = targets.cuda()

    if inputs.size()[0] is not hparams['batch_size']:
      continue

    # (3) forward pass
    likelihood = model(inputs)

    # (4) loss calculation
    loss = loss_fn(likelihood, targets)
    # add to epoch loss
    epoch_loss += loss.item()
    # print(loss.item())

    # (5) optimization
    loss.backward()
    clip_gradient(model, 1e-1)
    optim.step()
    steps += 1
    epoch_loss += loss.item()

    num_corrects = (torch.max(likelihood, 1)[1].view(targets.size()).data == targets.data).float().sum()
    acc = 100.0 * num_corrects/len(batch[0])
    epoch_accuracy += acc

    # if idx and idx%100 == 0:
    #  print('({}) Iteration loss : {}'.format(idx, loss.item()))

  print('Epoch loss : {}, Epoch accuracy : {}%'.format(epoch_loss/steps, epoch_accuracy/steps))

  return epoch_loss/steps, epoch_accuracy/steps


def evaluate(model, test_iter, hparams):
  epoch_loss, epoch_accuracy = 0., 0.
  loss_fn = hparams['loss_fn']

  # prepare model for evaluation
  model.eval()
  if torch.cuda.is_available():
    model.cuda()

  steps = 0
  with torch.no_grad():
    for idx, batch in enumerate(test_iter):

      # (1) get inputs and targets
      inputs, targets = batch

      # if cuda
      if torch.cuda.is_available():
        inputs = inputs.cuda()
        targets = targets.cuda()

      if inputs.size()[0] is not 32:
        continue

      # (2) forward
      likelihood = model(inputs)

      # (3) loss calc
      loss = loss_fn(likelihood, targets)
      epoch_loss += loss.item()

      # (4) accuracy calc
      num_corrects = (torch.max(likelihood, 1)[1].view(targets.size()).data == targets.data).float().sum()
      acc = 100.0 * num_corrects / len(batch[0])
      epoch_accuracy += acc.item()

      steps += 1

    print('::[evaluation] Loss : {}, Accuracy : {}'.format(
      epoch_loss / (steps), epoch_accuracy / (steps)))

    return epoch_loss / steps, epoch_accuracy / steps


def training(model, hparams, train_iter, valid_iter, epochs=10):

  # NOTE select best parameters based on accuracy on validation set
  ev_accuracies = []
  for epoch in range(epochs):
    print('[{}]'.format(epoch+1))
    tr_loss, tr_accuracy = train_epoch(model, train_iter, hparams)
    ev_loss, ev_accuracy = evaluate(model, valid_iter, hparams)

    # check for best parameters criterion
    if len(ev_accuracies) and ev_accuracy > max(ev_accuracies):
      torch.save(model, MODEL_SAVE_FILE)

    # keep track of evaluation accuracy
    ev_accuracies.append(ev_loss)


if __name__ == '__main__':
  # get data
  if args.data == 'aclImdb':
    (train_iter, test_iter), vocab, w2i = preprocessed_aclImdb()
  else:
    (train_iter, test_iter), vocab, w2i = preprocessed_socialmedia()

  # define a loss function
  loss_fn = F.cross_entropy

  # set hyperparameters
  hparams = {
    'vocab_size'  : len(vocab),
    'emb_dim'     : args.embed_dim,
    'hidden_dim'  : args.hidden_dim,
    'lr'          : 1e-3,
    'output_size' : 2,
    'loss_fn'     : loss_fn,
    'batch_size'  : args.batch_size
    }

  # create LSTM model
  lstmClassifier = LstmClassifier(hparams)

  # train model
  training(lstmClassifier, hparams, train_iter, test_iter, epochs=10)
