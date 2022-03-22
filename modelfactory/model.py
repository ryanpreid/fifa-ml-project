import abc
from helpers import training
from models import mlpreduced


class Model(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def load_model(self):
        pass

    @abc.abstractmethod
    def prediction(self):
        pass


class ReducedMLP(Model):
    def __init__(self):
        self.input_dim = 6
        self.output_dim = 1

        self.model_type = mlpreduced.MLPReducedModel(self.input_dim, self.output_dim)

        self.model_name = "MLPRmodel-V1.pt"
        self.model_path = "trained_models"

        self.model = self.load_model()

    def load_model(self):
        return training.get_saved_model(self.model_type, self.model_name, self.model_path)

    def prediction(self, tensor):

        predictions = self.model(tensor)

        return predictions.item()



    # TODO add different models. The idea is different frames can switch models or have their own.
    # TODO need to train other models first...
class ModelFactory:
    def get_model(self, name):
        if name == "ReducedMLP":
            return ReducedMLP()


