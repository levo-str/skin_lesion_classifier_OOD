from timeit import default_timer as timer

import torch.nn as nn
import torch

def test_model(model, test_loader, device):
    """
    Test a model on the test data
    :param model: the model to test
    :param test_loader: data loader for the test dataset
    :param device: device on which to perform the computations
    :return: accuracy and loss of the model on the test dataset
    """
    test_loss = 0
    num_correct_preds_valid = 0
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    ######################
    # evaluate the model #
    ######################
    model.eval()  # prep model for evaluation
    print("###################\n# test the model # \n###################")
    with torch.no_grad():
        for images, disease_labels in test_loader:
            input = images.float().to(device)
            labels = disease_labels.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(input)
            # calculate the loss
            loss = criterion(output, labels)
            # update running validation loss
            test_loss += loss.item() * labels.size(0)
            # update the number of correct predictions
            num_correct_preds_valid += (output.softmax(-1).argmax(-1) == labels).sum()

    # training/validation statistics
    test_loss = test_loss / len(test_loader.sampler)
    test_acc = num_correct_preds_valid / len(test_loader.sampler)
    print(f"Test Loss: {test_loss} \tTest Accuracy: {test_acc}")
    return test_loss, test_acc
