import pandas as pd
import torch
from torch.utils.data import Dataset
# Defining a Dataset class
#     To create a custom Dataset class, there are a few things you need to do:
#
# Load in a saved dataset file, typically as a DataFrame
# Split the DataFrame into input features and target(s)
# Convert these features and targets into tensor types
# Implement the Dataset superclass functions: __init__, __len__,
# and __getitem__ (which should return a tuple of input features and a target)


class FifaDataset(Dataset):
    def __init__(self, cvs_file_path):

        self.cvs_file_path = cvs_file_path

        df = pd.read_csv(self.cvs_file_path, index_col=False)

        input_features = df.iloc[:, :-1].values
        target = df.iloc[:, -1:].values

        self.x = torch.tensor(input_features, dtype=torch.float32)
        self.y = torch.tensor(target, dtype=torch.float32)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)
