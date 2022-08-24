import torch
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        # Convolution block
        self.conv_layer = nn.Sequential(
            nn.Conv1d(1, 8, 20, stride=10, padding=5),
            nn.ReLU(),
            nn.Conv1d(8, 16, 8, stride=4, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        )

        # Fully connected block
        self.fc_layer = nn.Sequential(
            nn.Linear(16 * 42, 300),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(300, 5)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(-1, 16 * 42)
        x = self.fc_layer(x)
        return x