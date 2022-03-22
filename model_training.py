import torch
from helpers import training
from models import mlpreduced
from pytorchdataset import fifadataset
from torch.nn import MSELoss
from torch.utils.data import DataLoader

torch.manual_seed(0)
torch.set_printoptions(threshold=10000)

training_path = "/Users/ryanreid/Dev/fifa-ml-project/data/prepared_outfield_player_data_reduced/training_outfield.csv"
test_path = "/Users/ryanreid/Dev/fifa-ml-project/data/prepared_outfield_player_data_reduced/test_outfield.csv"
valid_path = "/Users/ryanreid/Dev/fifa-ml-project/data/prepared_outfield_player_data_reduced/valid_outfield.csv"

training_dataset = fifadataset.FifaDataset(training_path)
test_dataset = fifadataset.FifaDataset(test_path)
valid_dataset = fifadataset.FifaDataset(valid_path)

training_loader = DataLoader(training_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Model variables
input_dim = 6
output_dim = 1

MLPRModel = mlpreduced.MLPReducedModel(input_dim, output_dim)

# Hyperparameters
criterion = MSELoss()
optimizer = torch.optim.SGD(MLPRModel.parameters(), lr=0.00001)

epochs = 5000
model_name = "MLPRmodel-V1.pt"
model_path = "trained_models"

model = training.fifa_training_validation_loop(epochs,
                                               training_loader,
                                               test_loader,
                                               MLPRModel,
                                               optimizer,
                                               criterion,
                                               model_name,
                                               model_path,
                                               True)

training.fifa_test_loop(test_loader, criterion, model)
