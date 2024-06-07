import sys

import numpy as np
import torch
from torch import nn
import scipy.stats as st


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def get_monte_carlo_predictions(data_loader,
                                nb_forward_passes,
                                model,
                                n_classes,
                                n_samples, device="cuda"):
    """ Function to get the monte-carlo samples and uncertainty estimates
    through multiple forward passes

    Parameters
    ----------
    data_loader : object
        data loader object from the data loader module
    nb_forward_passes : int
        number of monte-carlo samples/forward passes
    model : object
        keras model
    n_classes : int
        number of classes in the dataset
    n_samples : int
        number of samples in the test set
    """

    dropout_predictions = np.empty((0, n_samples, n_classes))
    for i in range(nb_forward_passes):
        predictions = np.empty((0, n_classes))
        model.eval()
        enable_dropout(model)
        for i, (image, label) in enumerate(data_loader):
            image = image.float().to(torch.device(device))
            with torch.no_grad():
                output = model(image)
                predictions = np.vstack((predictions, output.cpu().numpy()))

        dropout_predictions = np.vstack((dropout_predictions,
                                         predictions[np.newaxis, :, :]))
        # dropout predictions - shape (forward_passes, n_samples, n_classes)

    # Calculating mean across multiple MCD forward passes
    mean = np.mean(dropout_predictions, axis=0)  # shape (n_samples, n_classes)

    # Calculating variance across multiple MCD forward passes
    variance = np.var(dropout_predictions, axis=0)  # shape (n_samples, n_classes)

    epsilon = sys.float_info.min
    # Calculating entropy across multiple MCD forward passes
    entropy = -np.sum(mean * np.log(mean + epsilon), axis=-1)  # shape (n_samples,)

    # Calculating mutual information across multiple MCD forward passes
    mutual_info = entropy - np.mean(np.sum(-dropout_predictions * np.log(dropout_predictions + epsilon),
                                           axis=-1), axis=0)  # shape (n_samples,)

    return mean, variance, entropy, mutual_info


def get_monte_carlo_predictions_single_image(img, nb_forward_passes,
                                             model,
                                             n_classes):
    """ Function to get the monte-carlo samples and uncertainty estimates
        through multiple forward passes for one image

        Parameters
        ----------
        img : object
        image whose monte carlo prediction we want to calculate
        nb_forward_passes : int
            number of monte-carlo samples/forward passes
        model : object
            keras model
        n_classes : int
            number of classes in the dataset
        """
    predictions = np.empty((0, n_classes))
    for i in range(nb_forward_passes):
        model.eval()
        enable_dropout(model)
        with torch.no_grad():
            output = model(img)
        predictions = np.vstack((predictions, output.cpu().numpy()))

    # Calculating mean across multiple MCD forward passes

    mean = np.mean(predictions, axis=0)  # shape (n_forward_passes, n_classes)

    # Calculating variance across multiple MCD forward passes

    variance = np.var(predictions, axis=0)  # shape (n_forward_passes, n_classes)

    epsilon = sys.float_info.min

    # Calculating entropy across multiple MCD forward passes

    entropy = -np.sum(mean * np.log(mean + epsilon), axis=-1)  # shape (n_samples,)

    # Calculating mutual information across multiple MCD forward passes

    mutual_info = entropy - np.mean(np.sum(-predictions * np.log(predictions + epsilon),
                                           axis=-1), axis=0)

    return mean, variance, entropy, mutual_info


def get_confidence_interval_mcd__using_bernouilli_count(probability, nb_forward_passes, confidence):
    """
    :param probability: probability of the prediction given by:
    counting how many times the class we say corresponds to the argmax of the softmax outpout and divide by the number
    of forward passes
    :param nb_forward_passes: number of forward passes in the monte carlo dropout prediction
    :param confidence: the confidence we want to have in the result
    :return: confidence interval over the prediction
    """
    uncertainty = st.norm.ppf((1 - confidence) / 2 + confidence) * np.sqrt(
        1 / (nb_forward_passes * probability * (1 - probability)))
    return [probability - uncertainty, probability + uncertainty]


def get_confidence_interval_monte_carlo_using_prediction_mean(class_prediction, nb_forward_passes,
                                                              prediction_class_variance, confidence):
    """
    :param class_prediction: mean of the prediction given by the monte carlo dropout for the class with the highest mean
    i.e. the one we use as a classification
    :param nb_forward_passes: number of forward passes in the monte carlo dropout prediction
    :param prediction_class_variance: variance of the prediction for the class with the highest mean prediction
    :param confidence: the confidence we want to have in the result
    :return: confidence interval over the prediction
    """
    uncertainty = st.norm.ppf((1 - confidence) / 2 + confidence) * np.sqrt(
        1 / (nb_forward_passes)) * prediction_class_variance
    return [class_prediction - uncertainty, class_prediction + uncertainty]
