import torch
from torch import nn

class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pooling = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 * 16 * 16, 128)
        self.output = nn.Linear(128, num_classes)

    def forward(self, x):
     x = self.conv1(x) #-> (32, 128, 128)
     x = self.pooling(x) #-> (32, 64, 64)
     x = self.relu(x) 

     x = self.conv2(x) # -> (64, 64, 64)
     x = self.pooling(x) #-> (64, 32, 32)
     x = self.relu(x)

     x = self.conv3(x) # -> (128, 32, 32)
     x = self.pooling(x) #-> (128, 16, 16)
     x = self.relu(x)


     x = self.flatten(x)
     x = self.linear(x)
     x = self.output(x)

     return x