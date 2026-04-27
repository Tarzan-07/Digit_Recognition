"""
This is a torch implementation of Dense Attention-CNN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth):
        super().__init__()

        self.l1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=growth, kernel_size=1),
            nn.BatchNorm2d(growth),
            nn.ReLU(),
            nn.Conv2d(in_channels=growth, out_channels=growth, kernel_size=3, padding=1)
        )

    def forward(self, x):
        new_feat = self.l1(x)
        return torch.cat([x, new_feat], dim=1)

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels//2, 1, 1)
        )

    def forward(self, x):
        attn = torch.sigmoid(self.conv(x))
        return x*attn

class DenseAttnBlock(nn.Module):
    def __init__(self, in_channels, growth, num_layers):
        super().__init__()

        self.mod_list = nn.ModuleList()
        self.attn_list = nn.ModuleList()
        channels = in_channels

        for _ in range(num_layers):
            self.mod_list.append(DenseLayer(channels, growth))
            channels += growth
            self.attn_list.append(SpatialAttention(channels))

        self.out_channels = channels

    def forward(self, x):
        for layer, attn in zip(self.mod_list, self.attn_list):
            x = layer(x)
            x = attn(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction):
        super().__init__()

        self.fc1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels=in_channels//reduction, out_channels=in_channels, kernel_size=1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.relu(self.fc1(w))
        w = torch.softmax(self.fc2(w), dim=1)
        return x * w

class DACNN(nn.Module):
    def __init__(self, num_layers, num_classes=10):
        super().__init__()

        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.block1 = DenseAttnBlock(in_channels=32, growth=16, num_layers=num_layers)
        self.block2 = DenseAttnBlock(in_channels=self.block1.out_channels, growth=16, num_layers=num_layers)
        self.block3 = DenseAttnBlock(in_channels=self.block2.out_channels, growth=16, num_layers=num_layers)

        self.channel_attn = ChannelAttention(self.block3.out_channels, reduction=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.block3.out_channels, num_classes)
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.init_conv(x)

        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)

        x = self.channel_attn(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)

        x = self.dropout(x)
        x = self.fc(x)
        return x
