import matplotlib.pyplot as plt
import os
import torch

# TODO will need to refactor this a bit better. I.e. move the saving and loading into its static? class.
# TODO And move training related helpers into its own. Will keep this in the mean time.

def fifa_training_validation_loop(n_epochs, training_loader, valid_loader, model, optimizer, criterion, model_name,
                                  model_dir, plot):
    train_loss_list = []
    valid_loss_list = []

    print("starting training")
    for epoch in range(1, n_epochs + 1):
        train_loss = 0.0
        valid_loss = 0.0

        model.train()

        for inputs, target in training_loader:
            # clear the gradients
            optimizer.zero_grad()

            # make predictions from our inputs
            predictions = model(inputs)

            # calculate the loss
            loss = criterion(predictions, target)

            # propagate the loss backwards
            loss.backward()

            # update the weights and take a step with our optimizer.
            optimizer.step()

            # keep a track of our loss
            train_loss += loss.item()

        # Stop training model for validation
        model.eval()

        for inputs, target in valid_loader:
            predictions = model(inputs)

            loss = criterion(predictions, target)

            valid_loss += loss.item()

        train_loss_list.append(train_loss / len(training_loader))
        valid_loss_list.append(valid_loss / len(valid_loader))

        print('Epoch: {} \t Training Loss: {:.6f}, \t Validation Loss: {:.6f}'.format(epoch,
                                                                                      train_loss / len(training_loader),
                                                                                      valid_loss / len(valid_loader)))

    if plot:
        plt.figure(figsize=(8, 6))
        x = range(n_epochs)
        plt.plot(x, train_loss_list, label='train_loss')
        plt.plot(x, valid_loss_list, label='valid_loss')
        plt.legend()
        plt.show()

    print('saving model')
    save_model(model, model_name, model_dir)

    return model


# https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
def fifa_test_loop(data_loader, criterion, model):
    test_loss = 0
    correct = 0
    model.eval()

    # accuracy_list = []

    for inputs, target in data_loader:
        predictions = model(inputs)

        loss = criterion(predictions, target)

        test_loss += loss.item()

        # calculate and print avg test loss
        test_loss = test_loss / len(data_loader)

        # This is to make the values match close to targets
        rounded_predictions = torch.round(predictions)

        correct += (rounded_predictions == target).sum().item()
        correct /= rounded_predictions.size(0)

        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    # accuracy_list.append(100*correct)
    # plt.figure(figsize=(8,6))
    # x = range(test_epochs)
    # plt.plot(x, accuracy_list, label = 'accuracy_loss')
    # plt.legend()
    # plt.show()


def fifa_test_loop_load_model(data_loader, criterion, model, model_name, model_dir):
    test_loss = 0
    correct = 0
    model = get_saved_model(model, model_name, model_dir)
    model.eval()

    for inputs, target in data_loader:
        predictions = model(inputs)

        loss = criterion(predictions, target)

        test_loss += loss.item()

        # calculate and print avg test loss
        test_loss = test_loss / len(data_loader)

        # This is to make the values match close to targets
        rounded_predictions = torch.round(predictions)

        correct += (rounded_predictions == target).sum().item()
        correct /= rounded_predictions.size(0)

        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def save_model(model, model_name, model_dir):
    print("Saving the model as " + model_name)

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    path = os.path.join(model_dir, model_name)
    # saving state dictionary
    torch.save(model.state_dict(), path)


def get_saved_model(model, model_name, model_dir):
    print("looking for the following model: " + model_name)

    if not os.path.isdir(model_dir):
        Exception("Directory does not exist")

    path = os.path.join(model_dir, model_name)

    _model = model
    _model.load_state_dict(torch.load(path))

    return _model
