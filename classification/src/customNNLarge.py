import torch.nn.functional as f
import torch.nn as nn

INPUT_SHAPE = (3, 32, 32)
NUM_CLASSES = 2

class customNNLarge(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, input_shape=INPUT_SHAPE):
        super(customNNLarge, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, stride=1, padding=0)
        self.norm1 = nn.BatchNorm2d(32)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.norm2 = nn.BatchNorm2d(64)
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.norm3 = nn.BatchNorm2d(64)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(1024, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = f.relu(self.norm1(self.conv1(x)))
        x = self.pool1(x)
        x = f.relu(self.norm2(self.conv2(x)))
        x = self.pool2(x)
        x = f.relu(self.norm3(self.conv3(x)))
        x = self.flatten(x)
        x = f.relu(self.fc1(x))
        x = self.fc2(x)

        return x
