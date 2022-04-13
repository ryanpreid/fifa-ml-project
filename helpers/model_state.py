import os
import torch


def save_model(model, model_name, model_path):
    print("Saving the model as " + model_name)

    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    path = os.path.join(model_path, model_name)
    # saving state dictionary
    torch.save(model.state_dict(), path)


def get_saved_model(model, model_name, model_path):
    print("looking for the following model: " + model_name)

    if not os.path.isdir(model_path):
        Exception("Directory does not exist")

    path = os.path.join(model_path, model_name)

    _model = model
    _model.load_state_dict(torch.load(path))

    return _model