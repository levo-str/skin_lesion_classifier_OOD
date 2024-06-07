import os
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def train_model(model, train_loader, valid_loader, n_epochs, early_stopping_window, device, lr=0.001,
                freeze: bool = True,
                last_module_name='fc',
                save_checkpoints: bool = False, eps=0.01,
                save_checkpoints_path='/content/drive/MyDrive/OOD_2/mobilenet_checkpoint/',
                save_figures_path='/content/drive/MyDrive/OOD_2/mobilenet_checkpoint/',
                save_model_path='/content/drive/MyDrive/OOD_2/mobilenet_checkpoint/',
                checkpoint_path=None):
    # specify loss function (categorical cross-entropy)
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    # specify optimizer (stochastic gradient descent) and learning rate = 0.01
    if freeze:
        for name, param in model.named_parameters():
            if param.requires_grad and not last_module_name in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    non_frozen_parameters = [p for p in model.parameters() if p.requires_grad]
    # print(len(non_frozen_parameters))

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001
    optimizer = torch.optim.Adam(non_frozen_parameters, lr=lr)
    num_epochs_without_val_loss_reduction = 0
    valid_loss_min = np.inf
    start_epoch = 0
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        loss = checkpoint['loss']
    epochs = []
    train_accuracies = []
    valid_accuracies = []
    train_losses = []
    valid_losses = []
    for epoch in range(start_epoch, start_epoch + n_epochs):
        epochs.append(epoch)
        start_time = timer()
        # monitor losses
        train_loss = 0
        num_correct_preds_train = 0
        valid_loss = 0
        num_correct_preds_valid = 0

        ###################
        # train the model #
        ###################
        model.train()  # prep model for training
        print("###################\n# training the model # \n###################")
        for images, disease_labels in train_loader:
            input = images.float().to(device)
            labels = disease_labels.to(device)
            """ data = data.to(device)
            label = label.to(device) """
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(input)
            # calculate the loss
            loss = criterion(output, labels)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            # prevent introducing errors by varying batch sizes by multiplying
            train_loss += loss.item() * labels.size(0)
            # update the number of correct predictions
            num_correct_preds_train += (output.softmax(-1).argmax(-1) == labels).sum().item()

        train_loss = train_loss / len(train_loader.sampler)
        train_acc = num_correct_preds_train / len(train_loader.sampler)
        print('Epoch: {} \tTraining Loss: {:.6f} \tTrain Accuracy: {:.4f}'.format(
            epoch + 1,
            train_loss,
            train_acc,
        ))
        train_accuracies.append(train_acc)
        train_losses.append(train_loss)
        ######################
        # validate the model #
        ######################
        model.eval()  # prep model for evaluation
        print("###################\n# validate the model # \n###################")
        with torch.no_grad():
            for images, disease_labels in valid_loader:
                input = images.float().to(device)
                labels = disease_labels.to(device)
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(input)
                # calculate the loss
                loss = criterion(output, labels)
                # update running validation loss
                valid_loss += loss.item() * labels.size(0)
                # update the number of correct predictions
                num_correct_preds_valid += (output.softmax(-1).argmax(-1) == labels).sum().item()

        # print training/validation statistics
        # calculate average loss and accuracy over an epoch

        valid_loss = valid_loss / len(valid_loader.sampler)
        valid_acc = num_correct_preds_valid / len(valid_loader.sampler)

        print('Epoch: {} \tValidation Loss: {:.6f} \tValidation Accuracy: {:.4f}'.format(
            epoch + 1,
            valid_loss,
            valid_acc,
        ))
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_acc)
        if save_checkpoints:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(save_checkpoints_path, f'checkpoint_{epoch + 1}.pth'))
        end_time = timer()
        print(f"epoch : {epoch + 1} | time elapsed : {end_time - start_time}")
        # save model if validation loss has decreased
        if valid_loss - valid_loss_min < -eps:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            torch.save(model.state_dict(), os.path.join(save_model_path, f'final_model.pth'))
            valid_loss_min = valid_loss
            # reset the counter of epochs without validation loss reduction
            num_epochs_without_val_loss_reduction = 0
        else:
            num_epochs_without_val_loss_reduction += 1

        if num_epochs_without_val_loss_reduction >= early_stopping_window:
            # if we haven't had a reduction in validation loss for `early_stopping_window` epochs, then stop trainingz
            print(f'No reduction in validation loss for {early_stopping_window} epochs. Stopping training...')
            plt.plot(epochs, train_accuracies, label='train_accuracy')
            plt.plot(epochs, valid_accuracies, label='valid_accuracy')
            plt.savefig(save_figures_path + 'accuracy.png')
            plt.legend()
            plt.show()
            plt.plot(epochs, train_losses, label='train_loss')
            plt.plot(epochs, valid_losses, label='valid_loss')
            plt.legend()
            plt.show()
            plt.savefig(save_figures_path + 'loss.png')
            break


    plt.plot(epochs, train_accuracies, label='train_accuracy')
    plt.plot(epochs, valid_accuracies, label='valid_accuracy')
    plt.legend()
    plt.savefig(save_figures_path + 'accuracy.png')
    plt.show()
    plt.plot(epochs, train_losses, label='train_loss')
    plt.plot(epochs, valid_losses, label='valid_loss')
    plt.legend()
    plt.savefig(save_figures_path + 'loss.png')
    plt.show()
    return
