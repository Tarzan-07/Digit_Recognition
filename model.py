import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class SVNH(nn.Module):
    def __init__(self, num_layers, kernel_size, padding):
        super(SVNH, self).__init__()

        self.num_layer = num_layers
        self.kernel_size = kernel_size
        self.padding = padding
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=self.kernel_size, padding=self.padding)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=self.kernel_size, padding=self.padding)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=self.kernel_size, padding=self.padding)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(128*8*8, 512)
        self.fc2 = nn.Linear(512, 10)

        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        x = self.dropout(x)
        
        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x