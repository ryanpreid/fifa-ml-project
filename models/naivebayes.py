import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB


class NaiveBayes:
    def __init__(self):

        # Scaler object to be used an all data
        self.ms = MinMaxScaler()
        # Sort the data
        self.train_dataframe, self.train_target, self.test_dataframe, self.test_target = self.prepare_data()
        self.nb_model = self.naive_bayes_training()

    def prepare_data(self):

        # load the data
        training_path = "/Users/ryanreid/Dev/fifa-ml-project/data/k_means_data/training_outfield.csv"
        test_path = "/Users/ryanreid/Dev/fifa-ml-project/data/k_means_data/test_outfield.csv"

        # pass into dataframe
        train_df = pd.read_csv(training_path, index_col=False)
        test_df = pd.read_csv(test_path, index_col=False)

        # Unsupervised algorithm, so lets drop the target label
        train_X = train_df.drop(columns=['position_label'])
        train_Y = train_df['position_label']

        test_X = test_df.drop(columns=['position_label'])
        test_Y = test_df['position_label']

        # Feature scale the training data as it has been getting better accuracy.
        cols = train_X.columns
        train_X_scaled = self.ms.fit_transform(train_X.values)

        # Feature scale the test data as it has been getting better accuracy.
        test_X_scaled = self.ms.transform(test_X.values)

        # Convert data back to dataframe
        train_X_scaled_df = pd.DataFrame(train_X_scaled, columns=[cols])
        test_X_scaled_df = pd.DataFrame(test_X_scaled, columns=[cols])

        # Return scaled df and labels.
        return train_X_scaled_df, train_Y,  test_X_scaled_df, test_Y

    def naive_bayes_training(self):

        nb_model = GaussianNB()
        nb_model.fit(self.train_dataframe, self.train_target)

        return nb_model

    def accuracy(self):
        y_pred = self.nb_model.predict(self.test_dataframe.values)
        print("Accuracy:", metrics.accuracy_score(self.test_target, y_pred))

    def get_model(self):
        return self.nb_model

    def nb_prediction(self, slider_input):

        np_input = np.array([slider_input])
        input_scaled = self.ms.transform(np_input)

        # Scaling the input is triggering some values to be negative.
        # This is affecting the prediction calculations
        # So i can set the negatives to zero.
        # As a result, getting better predictions from input
        input_scaled[input_scaled <= 0] = 0

        return self.nb_model.predict(input_scaled)


