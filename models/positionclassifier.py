import torch.nn as nn
import torch.nn.functional as F


# Taking notes of the hyperparameters I used
# Hyperparameters
# criterion = BCEWithLogitsLoss()
# optimizer = torch.optim.SGD(classification_model.parameters(), lr=0.001,  weight_decay=0.001)
# epochs = 2000
# model_name = "Classification_Model-V8_regularization.pt"
class PositionClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PositionClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 250)
        self.fc2 = nn.Linear(250, 50)
        self.fc3 = nn.Linear(50, output_dim)

    # No need to use Sigmoid for activation
    # Loss function automatically applies it.
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
