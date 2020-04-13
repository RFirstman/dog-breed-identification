import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):
    def __init__(self, im_size, hidden_dim, kernel_size, n_classes):
        '''
        Simple CNN for testing purposes.

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        channels, H, W = im_size

        self.conv = nn.Conv2d(channels, hidden_dim, kernel_size=kernel_size, stride=1, padding=2)
        self.ReLU1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.linear1 = nn.Linear(hidden_dim * 16 * 16, 64)
        self.ReLU2 = nn.ReLU()

        self.linear2 = nn.Linear(64, n_classes)
        self.softmax = nn.Softmax(dim=1)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################


    def forward(self, images):
        '''
        Take a batch of images and run them through the model to
        produce a score for each class.

        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width

        Returns:
            A torch Variable of size (N, n_classes) specifying the score
            for each example and category.
        '''
        scores = None
        #############################################################################
        # TODO: Implement the forward pass.
        #############################################################################
        out1 = self.pool(self.ReLU1(self.conv(images)))
        out1 = out1.view(images.size()[0], -1)
        out2 = self.ReLU2(self.linear1( out1 ))
        scores = self.softmax(self.linear2( out2 ))
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores

