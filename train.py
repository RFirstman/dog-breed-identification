import os
import argparse

import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn

from data.load_data import load_datasets

# Args
parser = argparse.ArgumentParser(description='Dog Breed Identification')
# Hyperparameters
parser.add_argument('--lr', type=float, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, metavar='M',
                    help='SGD momentum')
parser.add_argument('--weight-decay', type=float, default=0.0,
                    help='Weight decay hyperparameter')
parser.add_argument('--batch-size', type=int, metavar='N',
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--model',
                    choices=['softmax', 'convnet', 'twolayernn', 'mymodel'],
                    help='which model to train/evaluate')
parser.add_argument('--hidden-dim', type=int,
                    help='number of hidden features/activations')
parser.add_argument('--kernel-size', type=int,
                    help='size of convolution kernels/filters')
# Other configuration
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='number of batches between logging train status')
parser.add_argument('--root', defualt="./",
                    help="Root directory for project")
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
else:
    print("NOT using CUDA")

# Load Stanford Dogs Dataset using torch data paradigm
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# Stanford Dogs metadata
n_classes = 120

# Datasets
train_dataset, test_dataset, classes = load_datasets(args.root)

# DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset,
                 batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset,
                 batch_size=args.batch_size, shuffle=True, **kwargs)

# TODO: Load the model
model = None
if args.cuda:
    model.cuda()

# Set up loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = ptimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

def train(epoch):
    '''
    Train the model for one epoch
    '''

    model.train()

    # train loop
    for batch_idx, batch in enumerate(train_loader):
        # prepare data
        images, targets = Variable(batch[0]), Variable(batch[1])
        if args.cuda:
            images, targets = images.cuda(), targets.cuda()

        # Update parameters in model using optimizer
        scores = model.forward(images)
        loss = criterion(scores, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            val_loss, val_acc = evaluate('val', n_batches=4)
            train_loss = loss.data
            examples_this_epoch = batch_idx * len(images)
            epoch_progress = 100. * batch_idx / len(train_loader)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                  'Train Loss: {:.6f}\tVal Loss: {:.6f}\tVal Acc: {}'.format(
                epoch, examples_this_epoch, len(train_loader.dataset),
                epoch_progress, train_loss, val_loss, val_acc))

def evaluate(split, verbose=False, n_batches=None):
    '''
    Compute loss on val or test data.
    '''
    model.eval()
    loss = 0
    correct = 0
    n_examples = 0
    if split == 'val':
        loader = val_loader
    elif split == 'test':
        loader = test_loader
    for batch_i, batch in enumerate(loader):
        data, target = batch
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        loss += criterion(output, target, size_average=False).data
        # predict the argmax of the log-probabilities
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        n_examples += pred.size(0)
        if n_batches and (batch_i >= n_batches):
            break

    loss /= n_examples
    acc = 100. * correct / n_examples
    if verbose:
        print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            split, loss, correct, n_examples, acc))
    return loss, acc

for epoch in range(1, args.epochs + 1):
    train(epoch)
evaluate('test', verbose=True)

# Save the model (architecture and weights)
torch.save(model, args.model + '.pt')