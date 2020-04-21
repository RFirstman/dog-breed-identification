import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):

    def __init__(self):
        self.conv1 = nn.Conv2d(3, 32, 4)
        self.bnorm1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(p=0.1)

        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bnorm2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(p=0.2)

        self.conv3 = nn.Conv2d(64, 128, 2)
        self.bnorm3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(p=0.3)

        self.conv4 = nn.Conv2d(128, 256, 3)
        self.bnorm4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout(p=0.4)

        self.fc1 = nn.Linear(256*12*12, 1000)
        self.dropout5 = nn.Dropout(p = 0.5)
        self.fc2 = nn.Linear(1000, 1000)
        self.dropout6 = nn.Dropout(p = 0.6)
        self.fc3 = nn.Linear(1000, 250)
        self.dropout7 = nn.Dropout(p = 0.7)
        self.fc4 = nn.Linear(250, 120)

    def forward(self, x):
        # Convolutional Layers
        x = self.dropout1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        x = self.dropout3(self.pool3(F.relu(self.bn3(self.conv3(x)))))
        x = self.dropout4(self.pool4(F.relu(self.bn4(self.conv4(x)))))

        layer1 = self.relu1(self.bnorm1(self.conv1(x)))
        layer1 = self.dropout1(self.pool1(layer1))

        layer2 = self.relu2(self.bnorm2(self.conv2(layer1)))
        layer2 = self.dropout2(self.pool2(layer2))

        layer3 = self.relu3(self.bnorm3(self.conv3(layer2)))
        layer3 = self.dropout3(self.pool3(layer3))

        layer4 = self.relu4(self.bnorm4(self.conv4(layer3)))
        layer4 = self.dropout4(self.pool4(layer4))

        # flatten
        conv_out = layer4.view(x.size(0), -1)

        # Fully Connected Layers
        layer5 = self.dropout5(self.fc1(conv_out))
        layer6 = self.dropout6(self.fc2(layer5))
        layer7 = self.dropout7(self.fc3(layer6))
        layer8 = self.fc4(layer7)

        # convert the 120 outputs into a distribution of class scores
        out = F.log_softmax(layer8, dim=1)

        return out
