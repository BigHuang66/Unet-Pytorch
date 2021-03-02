
import numpy as np
import torch
import torch.nn.functional as F


def calculate_area(pred, label, num_classes, ignore_index=255):
    """
    Calculate intersect, prediction and label area

    Args:
        pred (Tensor): The prediction by model.
        label (Tensor): The ground truth of image.
        num_classes (int): The unique number of target classes.
        ignore_index (int): Specifies a target value that is ignored. Default: 255.

    Returns:
        Tensor: The intersection area of prediction and the ground on all class.
        Tensor: The prediction area on all class.
        Tensor: The ground truth area on all class
    """
    if len(pred.shape) == 4:
        pred = torch.squeeze(pred, axis=1)
    if len(label.shape) == 4:
        label = torch.squeeze(label, axis=1)
    if not pred.shape == label.shape:
        raise ValueError('Shape of `pred` and `label should be equal, '
                         'but there are {} and {}.'.format(
                             pred.shape, label.shape))

    # Delete ignore_index
    mask = label != ignore_index
    pred = pred + 1
    label = label + 1
    pred = pred * mask
    label = label * mask
    pred = F.one_hot(pred.long(), num_classes + 1)
    label = F.one_hot(label.long(), num_classes + 1)
    pred = pred[:, :, :, 1:]
    label = label[:, :, :, 1:]

    pred_area = []
    label_area = []
    intersect_area = []

    for i in range(num_classes):
        pred_i = pred[:, :, :, i]
        label_i = label[:, :, :, i]
        pred_area_i = torch.sum(pred_i)
        label_area_i = torch.sum(label_i)
        intersect_area_i = torch.sum(pred_i * label_i)
        pred_area.append(pred_area_i)
        label_area.append(label_area_i)
        intersect_area.append(intersect_area_i)
    pred_area = torch.stack(pred_area)
    label_area = torch.stack(label_area)
    intersect_area = torch.stack(intersect_area)
    return intersect_area, pred_area, label_area


def mean_iou(intersect_area, pred_area, label_area):
    """
    Calculate iou.

    Args:
        intersect_area (Tensor): The intersection area of prediction and ground truth on all classes.
        pred_area (Tensor): The prediction area on all classes.
        label_area (Tensor): The ground truth area on all classes.

    Returns:
        np.ndarray: iou on all classes.
        float: mean iou of all classes.
    """
    intersect_area = intersect_area.cpu().numpy()
    pred_area = pred_area.cpu().numpy()
    label_area = label_area.cpu().numpy()
    union = pred_area + label_area - intersect_area
    class_iou = []
    for i in range(len(intersect_area)):
        if union[i] == 0:
            iou = 0
        else:
            iou = intersect_area[i] / union[i]
        class_iou.append(iou)
    miou = np.mean(class_iou)
    return np.array(class_iou), miou

def dice(intersect_area, pred_area, label_area):
    """
    Calculate dice.

    Args:
        intersect_area (Tensor): The intersection area of prediction and ground truth on all classes.
        pred_area (Tensor): The prediction area on all classes.
        label_area (Tensor): The ground truth area on all classes.

    Returns:
        np.ndarray: dice on all classes.
        float: mean dice of all classes.
    """
    intersect_area = intersect_area.cpu().numpy()
    pred_area = pred_area.cpu().numpy()
    label_area = label_area.cpu().numpy()
    sum_area = pred_area + label_area
    class_dice = []
    for i in range(len(intersect_area)):
        if sum_area[i] == 0:
            dice = 0
        else:
            dice = 2*intersect_area[i] / sum_area[i]
        class_dice.append(dice)
    dice = np.mean(class_dice)
    return np.array(class_dice), dice



def accuracy(intersect_area, pred_area):
    """
    Calculate accuracy

    Args:
        intersect_area (Tensor): The intersection area of prediction and ground truth on all classes..
        pred_area (Tensor): The prediction area on all classes.

    Returns:
        np.ndarray: accuracy on all classes.
        float: mean accuracy.
    """
    intersect_area = intersect_area.cpu().numpy()
    pred_area = pred_area.cpu().numpy()
    class_acc = []
    for i in range(len(intersect_area)):
        if pred_area[i] == 0:
            acc = 0
        else:
            acc = intersect_area[i] / pred_area[i]
        class_acc.append(acc)
    macc = np.sum(intersect_area) / np.sum(pred_area)
    return np.array(class_acc), macc


def kappa(intersect_area, pred_area, label_area):
    """
    Calculate kappa coefficient

    Args:
        intersect_area (Tensor): The intersection area of prediction and ground truth on all classes..
        pred_area (Tensor): The prediction area on all classes.
        label_area (Tensor): The ground truth area on all classes.

    Returns:
        float: kappa coefficient.
    """
    intersect_area = intersect_area.cpu().numpy()
    pred_area = pred_area.cpu().numpy()
    label_area = label_area.cpu().numpy()
    total_area = np.sum(label_area)
    po = np.sum(intersect_area) / total_area
    pe = np.sum(pred_area * label_area) / (total_area * total_area)
    kappa = (po - pe) / (1 - pe)
    return kappa
