import numpy as np
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

# There is some magic numbers here.
# There are 13 input features and 4 classes for classification.
# https://stackoverflow.com/questions/50981714/multi-label-multi-class-image-classifier-convnet-with-pytorch


class FifaMultilabelDataset(Dataset):
    def __init__(self, cvs_file_path):

        self.cvs_file_path = cvs_file_path

        self.input_features_amount = 13
        self.target_classes = 4

        df = pd.read_csv(self.cvs_file_path, index_col=False)

        input_features = df.iloc[:, :-self.target_classes].values
        target = np.asarray(df.iloc[:, self.input_features_amount:])

        self.x = torch.tensor(input_features, dtype=torch.float32)
        self.y = torch.tensor(target, dtype=torch.float32)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


