import torch
import torch.nn as nn
import torch.nn.functional as F

class StegoDetector(nn.Module):
    def __init__(self):
        super(StegoDetector, self).__init__()
        self.conv1 = nn.Conv2d(24, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(256*8*8, 512) # Adjusted for input size 32x32
        self.fc2 = nn.Linear(512, 2)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StegoDetector().to(device)
print(model)

