import torch
from helpers import classifier_training, training
from models import positionclassifier
from pytorchdataset import fifamultilabeldataset
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader

torch.manual_seed(0)
torch.set_printoptions(threshold=10000)

training_path = "/Users/ryanreid/Dev/fifa-ml-project/data/multiclass_classification_data/training_players.csv"
test_path = "/Users/ryanreid/Dev/fifa-ml-project/data/multiclass_classification_data/test_players.csv"
valid_path = "/Users/ryanreid/Dev/fifa-ml-project/data/multiclass_classification_data/valid_players.csv"

training_dataset = fifamultilabeldataset.FifaMultilabelDataset(training_path)
test_dataset = fifamultilabeldataset.FifaMultilabelDataset(test_path)
valid_dataset = fifamultilabeldataset.FifaMultilabelDataset(valid_path)

training_loader = DataLoader(training_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Model variables
input_dim = 13
output_dim = 4

classification_model = positionclassifier.PositionClassifier(input_dim, output_dim)

# Hyperparameters
criterion = BCEWithLogitsLoss()
optimizer = torch.optim.SGD(classification_model.parameters(), lr=0.001, weight_decay=0.001)

epochs = 2000
model_name = "Classification_Model-V8_regularization.pt"
model_path = "trained_models"

model = training.training_and_validation_loop(epochs,
                                              training_loader,
                                              test_loader,
                                              classification_model,
                                              optimizer,
                                              criterion,
                                              model_name,
                                              model_path,
                                              True)

classifier_training.test_loop_for_classifier_with_loaded_model(test_loader, criterion, classification_model, model_name, model_path)
