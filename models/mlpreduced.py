import torch.nn as nn
import torch.nn.functional as F


class MLPReducedModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPReducedModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 48)
        self.fc2 = nn.Linear(48, 24)
        self.fc3 = nn.Linear(24, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x