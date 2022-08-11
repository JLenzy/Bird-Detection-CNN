# This was modified from Valerio Velardo's brilliaant tutorial
# https://youtu.be/SQ1iIKs190Q

from torch import nn
from torchsummary import summary

class CNNNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        # 4 convolutional blocks / flatten / linear / softmax

        # Grid size:  # of pixels for height/width  (odd number value)
        # Stride: Step size for sliding kernel across matrix
        # Depth: How many independent channels
        # i.e. for RGB image, it has a kernel of 3 x 3 x 3 because of RGB channels
        # Number of Kernels: Output from a layer has as many 2day arrays as # of kernels
        # ---------
        # Pooling: Shrinks the data
        # Parameters: Grid size, Stride and Type (max, average)
        # ---------
        # Calculating Data Shape:
        # 13 MFCCS, 512 sample hop length, 51200 samples in audio file
        # Shape = 100 x 13 x 1
        # 100 = (total samples / hop length)
        # 13 = num_MFCC


        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 * 5 * 4, 1)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        predictions = self.sigmoid(logits)
        #predictions = self.softmax(logits)
        return predictions

if __name__ == "__main__":
    cnn = CNNNetwork()
    summary(cnn, (1, 64, 44))

