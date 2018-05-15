"""
Code based loosely on implementation:
https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

Under MIT license.
"""
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    A simple MNIST classification model.

    Conv 3x3 - 32
    Conv 3x3 - 64
    MaxPool 2x2
    Dropout 0.25
    Flatten
    Dense - 128
    Dense - output (softmax)
    """

    def __init__(self, img_rows, img_cols, img_channels, num_classes):
        super(Net, self).__init__()

        self.flattened_size = (img_rows - 4) // 2 * (img_cols - 4) // 2 * 64

        self.conv1 = nn.Conv2d(in_channels=img_channels, out_channels=32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))

        # 1179648

        self.dropout1 = nn.Dropout2d(p=0.25)
        self.dropout2 = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=(2, 2))
        x = self.dropout1(x)
        x = x.view(-1, self.flattened_size)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def create(img_rows, img_cols, img_channels, num_classes):
    """ Create the model matching specified image dimensions """
    return Net(img_rows, img_cols, img_channels, num_classes)
