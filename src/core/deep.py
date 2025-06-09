from torch import nn

class HandDrawCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # features extraction
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (B, 32, 28, 28)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # (B, 32, 14, 14)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (B, 64, 14, 14)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # (B, 64, 7, 7)
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64*7 *7, out_features=512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layers(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
