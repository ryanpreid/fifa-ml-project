import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


class Kmeans:
    def __init__(self):

        # Sort the data
        self.dataframe, self.target = self.prepare_data()

        self.kmeans_model = None
        self.data_with_predicted_clusters = None

    def prepare_data(self):

        # load the data
        training_path = "/Users/ryanreid/Dev/fifa-ml-project/data/k_means_data/training_outfield.csv"

        # pass into dataframe
        train_df = pd.read_csv(training_path)

        # Unsupervised algorithm, so lets drop the target label
        X = train_df.drop(columns=['position_label'])
        Y = train_df['position_label']

        # Feature scale the data as it has been getting better accuracy.
        cols = X.columns
        ms = MinMaxScaler()
        X_scaled = ms.fit_transform(X)

        # Convert data back to dataframe
        X_scaled_df = pd.DataFrame(X_scaled, columns=[cols])

        # Return scaled df and labels.
        return X_scaled_df, Y

    def kmeans_training(self, clusters):

        kmeans_model = KMeans(n_clusters=clusters, init="random", max_iter=300, algorithm='auto', random_state=0)
        self.kmeans_model = kmeans_model

    def kmeans_prediction(self):

        if self.kmeans_model is None:
            raise Exception("kmeans model has not been trained. Please perform a training pass")

        predictions = self.kmeans_model.fit_predict(self.dataframe)

        data_with_clusters = self.dataframe.copy()
        data_with_clusters['clusters'] = predictions

        self.data_with_predicted_clusters = data_with_clusters

    def get_accuracy(self):

        labels = self.kmeans_model.labels_
        correct_labels = sum(self.target == labels)
        print("******ACCCURACY*******")
        print(correct_labels, self.target.size)

        accuracy = ('Accuracy score: {0:0.2f}'. format(correct_labels/float(self.target.size)))
        print(accuracy)

        return accuracy

    def get_2d_plot(self, figure_axis):

        axis = figure_axis
        axis.cla()
        axis.scatter(self.data_with_predicted_clusters["attacking_overall"], self.data_with_predicted_clusters["defending_overall"], c=self.data_with_predicted_clusters["clusters"])
        axis.scatter(self.kmeans_model.cluster_centers_[:, 1],self.kmeans_model.cluster_centers_[:, 2], s=300, c='red')
        axis.set_xlabel("Attacking overall")
        axis.set_ylabel("defending overall")
        return axis

    def get_3d_plot(self, axes):

        ax = axes
        ax.cla()
        ax.scatter(self.kmeans_model.cluster_centers_[:, 1], self.kmeans_model.cluster_centers_[:, 2], self.kmeans_model.cluster_centers_[:, -1], s=500, c='black', zorder=10)
        ax.scatter(self.data_with_predicted_clusters['attacking_overall'], self.data_with_predicted_clusters['defending_overall'], self.data_with_predicted_clusters['goalkeeping_overall'] , c=self.data_with_predicted_clusters["clusters"],
               cmap="rainbow", marker='o', zorder=1)

    def elbow_method(self):
        cs = []
        for i in range (1, 11):
            kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init=10, random_state=0)
            kmeans.fit(self.dataframe)
            cs.append(kmeans.inertia_)
        plt.plot(range(1, 11), cs)
        plt.title('The Elbow Method')
        plt.xlabel("number of clusters")
        plt.ylabel('CS')
        plt.show()
