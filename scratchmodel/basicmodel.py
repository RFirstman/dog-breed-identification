import torch
import torch.nn as nn


class BasicCNN(nn.Module):
  def __init__(self, num_classes):
    super(BasicCNN, self).__init__()
    channels_in = 3 # RGB Image
    self.conv_layer1 = nn.Sequential(
        nn.Conv2d(3, 32, 4),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(2, 2),
        nn.Dropout(p=0.1)
    )
    self.conv_layer2 = nn.Sequential(
        nn.Conv2d(32, 64, 3),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(2, 2),
        nn.Dropout(p=0.1)
    )
    self.conv_layer3 = nn.Sequential(
        nn.Conv2d(64, 128, 3),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(2, 2),
        nn.Dropout(p=0.1)
    )
    self.max_pool1 = nn.MaxPool2d(2, 2)
    self.conv_layer4 = nn.Sequential(
        nn.Conv2d(128, 256, 3),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(2, 2),
        nn.Dropout(p=0.1)
    )
    self.max_pool2 = nn.MaxPool2d(2, 2)

    self.fc1 = nn.Linear(1024, 512)
    self.fc2 = nn.Linear(512, num_classes)

  def forward(self, x):
    # print(x.size())

    x = self.conv_layer1(x)
    x = self.conv_layer2(x)
    x = self.conv_layer3(x)
    x = self.max_pool1(x)
    x = self.conv_layer4(x)
    x = self.max_pool2(x)
    
    x = x.view(x.size(0), -1)
    out = self.fc2(self.fc1(x))

    return out