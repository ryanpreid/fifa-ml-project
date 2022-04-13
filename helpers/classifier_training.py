import numpy as np
import torch
from helpers import model_state


# https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
def test_loop_for_classifier_with_model(data_loader, criterion, model):
    test_loss = 0
    total = 0
    correct = 0
    model.eval()

    for inputs, target in data_loader:
        predictions = model(inputs)

        loss = criterion(predictions, target)

        test_loss += loss.item()

        # calculate and print avg test loss
        test_loss = test_loss / len(data_loader)

        total += target.size(0)
        correct += classifier_correct(predictions, target)

        accuracy = 100 * correct//total

        print(f"Test Error: \n Accuracy: {accuracy}%, Avg loss: {test_loss:>8f} \n")


def test_loop_for_classifier_with_loaded_model(data_loader, criterion, model, model_name, model_dir):
    test_loss = 0
    correct = 0
    total = 0
    model = model_state.get_saved_model(model, model_name, model_dir)
    model.eval()

    for inputs, target in data_loader:
        predictions = model(inputs)

        loss = criterion(predictions, target)

        test_loss += loss.item()

        # calculate and print avg test loss
        test_loss = test_loss / len(data_loader)

        total += target.size(0)
        correct += classifier_correct(predictions, target)

        accuracy = 100 * correct//total

        print(f"Test Error: \n Accuracy: {accuracy}%, Avg loss: {test_loss:>8f} \n")


def one_hot_model_output(output):

    output = output.detach().numpy()

    idx = np.argmax(output, axis=-1)
    one_hot_vector = np.zeros(output.shape)
    one_hot_vector[np.arange(one_hot_vector.shape[0]), idx] = 1

    one_hot_vector_tensor = torch.from_numpy(one_hot_vector)

    return one_hot_vector_tensor


def binary_acc(predictions, target):
    # Round the result from the predictions aka make them 0||1
    rounded_predictions = torch.round(predictions)

    correct = (rounded_predictions == target).sum().float()
    correct /= rounded_predictions.size(0)

    accuracy = 100 * correct

    return accuracy


def classifier_correct(predictions, target):
    # this encapsulates the functionality of converting the models output to a one hot vector
    # It then compares the argmax indexes of predictions and target to provide how many were correct.

    one_hot_output = one_hot_model_output(predictions)
    return (one_hot_output.argmax(1) == target.argmax(1)).sum().item()