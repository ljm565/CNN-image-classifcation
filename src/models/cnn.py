import torch
import torch.nn as nn



class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        self.height = config.height
        self.width = config.width
        self.label = config.class_num
        self.color_channel = config.color_channel

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.color_channel, out_channels=32, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.fc = nn.Linear(64 * (self.height-12) * (self.width-12), self.label)


    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = torch.flatten(output, 1)
        output = self.fc(output)
        return output