import numpy as np
import pandas as pd
import scipy as stats
from scipy.stats import chi2
import bisect
import torch
import tqdm
from torch import nn

activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


def calculate_mean_variance_feature(input_data_loader, model, feature_size: int, hook_name,
                                    save_distribution_parameters: bool = False, filename_suffix="", device="cuda"):
    """
    Calculate mean and variance of the features' distribution
    :param input_data_loader: data loader for the training data used for the distribution
    :param model: model used to get the feature
    :param feature_size: dimensionality of the feature
    :param hook_name: name of the forward hook used to get the feature
    :param save_distribution_parameters: Save the parameters? 
    :param filename_suffix: suffix to add the file name can be interesting if we're calculating class
    wise mean and variance
    :param device: device on which to perform the operations
    :return: parameters of the features' distribution
    """
    model.eval()  # prep model for evaluation
    mean = torch.zeros(feature_size).to(device)
    covariance = torch.zeros(feature_size, feature_size).to(device)
    with torch.no_grad():
        print("Calculating distribution mean... \n")
        for i, (images, _) in enumerate(tqdm(input_data_loader)):
            input = images.float().to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(input)
            mean += activation[hook_name].sum(0).to(device) / len(input_data_loader.dataset)
        if save_distribution_parameters:
            np.save(f"{model.__class__.__name__}_mean" + filename_suffix, mean.cpu().numpy())
        print("Calculating distribution covariance... \n")
        for i, (images, _) in enumerate(tqdm(input_data_loader)):
            input = images.float().to(device)
            output = model(input)
            covariance += torch.matmul(torch.transpose((activation[hook_name] - mean), 0, 1),
                                       (activation[hook_name] - mean)) * 1 / (len(input_data_loader.dataset) - 1)
            # in case memory is limited divide by 1/len(input_data_loader.sampler) to be sure to not exceed float biggest number
        if save_distribution_parameters:
            np.save(f"{model.__class__.__name__}_covariance" + filename_suffix, covariance.cpu().numpy())
        return mean, covariance


def calculate_mahalanobis(y, mean, inv_covariance_mat):
    """
    Calculates the mahalanobis distance between y and a distribution
    :param y: feature to which we want the measure the mahalanobis distance to the distribution
    :param mean: mean of the distribution
    :param inv_covariance_mat: inverted covariance matrix of the distribution
    :return: mahalanobis distance between y and the distribution
    """
    left = torch.matmul(y - mean, inv_covariance_mat)
    mahal = torch.matmul(left, torch.transpose((y - mean), 0, 1))
    # print(torch.diagonal(mahal))
    return torch.diagonal(mahal)


def calculate_percentile_mahalanobis(input_data_loader, model, feature_size: int, hook_name, percentile,
                                     index_to_label_dict,
                                     mean=None, covariance=None, save_distribution_parameters: bool = False, filename_suffix="",
                                     device="cuda"):
    """
    Calculates the distances between the distribution of the features of the input data and the features of the input
    data, which allows to set a threshold for OOD detection
    E.g: if 95% of the training data has a distance to the training distribution of 800 then it 800 can be set as a
    threshold to say that anything above will be considered OOD.
    :param input_data_loader: data loader for the training data used for the distribution
    :param model: model used to get the feature
    :param feature_size: dimensionality of the feature
    :param hook_name: name of the forward hook used to get the feature
    :param percentile: Percentile of the training data that will be considered as in distribution, can be used as
    threshold for the OOD detection
    :param index_to_label_dict:
    :param mean: mean of the training data's feature distribution
    :param covariance: covariance of the training data's feature distribution
    :param save_distribution_parameters:
    :param filename_suffix:
    :param device:
    :return: the table of mahalanobis distances between the training data and its distribution AND threshold based
    """
    if mean is None or covariance is None:
        mean, covariance = calculate_mean_variance_feature(input_data_loader, model, feature_size,
                                                           hook_name, save_distribution_parameters, filename_suffix, device)
    distances = []
    labels = []
    # instead of list store in data frame
    model.eval()
    model.to(device)
    with torch.no_grad():
        print("Calculating samples distance to distribution... \n")
        for i, (images, ground_truth_labels) in enumerate(tqdm(input_data_loader)):
            input = images.float().to(device)
            # TODO: check if there isn't a matrix way to do that.
            for i in range(input.size(0)):
                output = model(input[i].unsqueeze(0))
                # bisect.insort(distances, calculate_mahalanobis(activation[hook_name].cpu().numpy(), mean, covariance))
                labels.append(index_to_label_dict[ground_truth_labels[i].item()])
                inv_covariance_mat = torch.linalg.inv(covariance).to(device)
                distances.append(
                    calculate_mahalanobis(activation[hook_name].to(device), mean.to(device), inv_covariance_mat).item())
    result = pd.DataFrame({"label": labels, "distance": distances}, index=np.arange(0, len(labels)))
    result.sort_values(by="distance", inplace=True)
    return result, result.iloc[int(percentile * result.shape[0])]["distance"]


def predict_with_mahalanobis_detection(model, input, max_tolerable_distance, mean, inv_covariance_mat, feature_name,
                                       device):
    model.eval()
    model.to(device)
    output = model(input)
    distance_from_distribution = calculate_mahalanobis(activation[feature_name].to(device), mean.to(device),
                                                       inv_covariance_mat.to(device))
    if distance_from_distribution.item() > max_tolerable_distance:
        return "OOD"
    else:
        return output


def test_model_with_OOD_detection(model, test_loader, max_tolerable_distance, mean, inv_covariance_mat, feature_name,
                                  device, display_results=True):
    test_loss = 0
    num_correct_preds_valid = 0
    number_samples_in_distribution = 0
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    ######################
    # evaluate the model #
    ######################
    model.eval()  # prep model for evaluation
    print("###################\n# test the model # \n###################")
    with torch.no_grad():
        for images, ground_truth_labels in test_loader:
            input = images.float().to(device)
            labels = ground_truth_labels.to(device)
            # TODO: check if there isn't a matrix way to do that.
            for i in range(input.size(0)):
                output = predict_with_mahalanobis_detection(model, input[i].unsqueeze(0), max_tolerable_distance, mean,
                                                            inv_covariance_mat, feature_name, device)

                if type(output) is not str:
                    # update the number of correct predictions
                    # change that because there could be a problem where they have the same class number but they mean different things because they come
                    # from different dataset
                    num_correct_preds_valid += output.softmax(-1).argmax(-1) == ground_truth_labels[i]
                    number_samples_in_distribution += 1
                    # # calculate the loss
                    # loss = criterion(output, labels[i].unsqueeze(0))
                    # # update running validation loss
                    # test_loss += loss.item()

    # training/validation statistics
    if number_samples_in_distribution != 0:
        test_loss = test_loss / number_samples_in_distribution
        test_acc = num_correct_preds_valid / number_samples_in_distribution
    else:
        test_loss = "100% OOD"
        test_acc = "100% OOD"
    if display_results:
        print(
            f"Test Loss: {test_loss} \tTest Accuracy: {test_acc} \tProportion of samples considered : {number_samples_in_distribution / len(test_loader.sampler)}, \tProportion of ood detected:  {(len(test_loader.sampler) - number_samples_in_distribution) / len(test_loader.sampler)}")
    return test_loss, test_acc


def predict_with_class_conditioned_mahalanobis_detection(model, input, max_tolerable_distances, means,
                                                         inv_covariance_mats, feature_name, device):
    model.eval()
    model.to(device)
    output = model(input)
    distances_from_distribution = []
    for i in range(len(means)):
        distances_from_distribution.append(
            calculate_mahalanobis(activation[feature_name].to(device), means[i].to(device),
                                  inv_covariance_mats[i]).cpu().numpy())
    if min(distances_from_distribution) > max_tolerable_distances[np.argmin(distances_from_distribution)]:
        return "OOD"
    else:
        return output


def test_model_with_class_conditionned_OOD_detection(model, test_loader, max_tolerable_distances, means,
                                                     inv_covariance_mats, feature_name, device, display_results=True):
    test_loss = 0
    num_correct_preds_valid = 0
    number_samples_in_distribution = 0
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    ######################
    # evaluate the model #
    ######################
    model.eval()  # prep model for evaluation
    print("###################\n# test the model # \n###################")
    with torch.no_grad():
        for images, ground_truth_labels in test_loader:
            input = images.float().to(device)
            labels = ground_truth_labels.to(device)
            # TODO: check if there isn't a matrix way to do that.
            for i in range(input.size(0)):
                output = predict_with_class_conditioned_mahalanobis_detection(model, input[i].unsqueeze(0),
                                                                              max_tolerable_distances, means,
                                                                              inv_covariance_mats, feature_name,
                                                                              device)

                if type(output) is not str:
                    # update the number of correct predictions
                    num_correct_preds_valid += output.softmax(-1).argmax(-1) == ground_truth_labels[i]
                    number_samples_in_distribution += 1
                    # # calculate the loss
                    # loss = criterion(output, labels[i].unsqueeze(0))
                    # # update running validation loss
                    # test_loss += loss.item()

    # training/validation statistics
    if number_samples_in_distribution != 0:
        test_loss = test_loss / number_samples_in_distribution
        test_acc = num_correct_preds_valid / number_samples_in_distribution
    else:
        test_loss = "100% OOD"
        test_acc = "100% OOD"
    if display_results:
        print(
            f"Test Loss: {test_loss} \tTest Accuracy: {test_acc} \tProportion of samples considered : {number_samples_in_distribution / len(test_loader.sampler)}, \tProportion of ood detected:  {(len(test_loader.sampler) - number_samples_in_distribution) / len(test_loader.sampler)}")
    return test_loss, test_acc
