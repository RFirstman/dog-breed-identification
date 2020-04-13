# Code partially sourced from https://github.com/zrsmithson/Stanford-dogs/blob/master/train.py
# Project is focused on evaluation and experimentation, so I did not want to reinvent the wheel here.
import argparse
import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable

from data.load_data import load_datasets

# Training settings
parser = argparse.ArgumentParser(description='CIFAR-10 Evaluation Script')
parser.add_argument('--model',
                    help='full path of model to evaluate')
# parser.add_argument('--test-dir', default='data',
#                     help='directory that contains test_images.npy file '
#                          '(downloaded automatically if necessary)')
parser.add_argument('--root', defualt="./",
                    help="Root directory for project")
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Load Stanford Dogs data using torch data paradigm
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

_, test_dataset, classes = load_datasets(args.root)

test_loader = torch.utils.data.DataLoader(test_dataset,
                 batch_size=args.batch_size, shuffle=True, **kwargs)

if os.path.exists(args.model):
    model = torch.load(args.model)
else:
    print("Specified model path does not exist")
    sys.exit(1)

if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()

def evaluate():
    '''
    Compute loss on test data
    '''
    model.eval()

    # initialize tensor and lists to monitor test loss and accuracy
    test_loss = torch.zeros(1).cuda()
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))

    for batch_i, batch in enumerate(test_loader):
        inputs, labels = batch
        if args.cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        # feedforward
        output = model(inputs)

        # calculate loss
        loss = criterion(output, labels)

        # average test loss
        test_loss = test_loss + ((torch.ones(1).cuda() / (batch_i + 1)) * (loss.data - test_loss))

        # get prediction
        _, predicted = torch.max(output.data, 1)

        # compare predictions to true label
        # this creates a `correct` Tensor that holds the number of correctly classified images in a batch
        correct = np.squeeze(predicted.eq(labels.data.view_as(predicted)))

        # calculate test accuracy for *each* object class
        # we get the scalar value of correct items for a class, by calling `correct[i].item()`
        for l, c in zip(labels.data, correct):
            class_correct[l] += c.item()
            class_total[l] += 1

        print('Test Loss: {:.6f}\n'.format(test_loss.cpu().numpy()[0]))

        for i in range(len(classes)):
            if class_total[i] > 0:
                print('Test Accuracy of %30s: %2d%% (%2d/%2d)' % (
                    classes[i], 100 * class_correct[i] / class_total[i],
                    np.sum(class_correct[i]), np.sum(class_total[i])))
            else:
                print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))


        print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
            100. * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct), np.sum(class_total)))
