
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils import metrics, Timer, calculate_eta, logger, progbar
from tools import infer

np.set_printoptions(suppress=True)


def evaluate(model,
             eval_dataset,
             device =None,
             aug_eval=False,
             scales=1.0,
             flip_horizontal=True,
             flip_vertical=False,
             is_slide=False,
             stride=None,
             crop_size=None,
             num_workers=0):
    """
    Launch evalution.

    Args:
        model（nn.Layer): A sementic segmentation model.
        eval_dataset (paddle.io.Dataset): Used to read and process validation datasets.
        aug_eval (bool, optional): Whether to use mulit-scales and flip augment for evaluation. Default: False.
        scales (list|float, optional): Scales for augment. It is valid when `aug_eval` is True. Default: 1.0.
        flip_horizontal (bool, optional): Whether to use flip horizontally augment. It is valid when `aug_eval` is True. Default: True.
        flip_vertical (bool, optional): Whether to use flip vertically augment. It is valid when `aug_eval` is True. Default: False.
        is_slide (bool, optional): Whether to evaluate by sliding window. Default: False.
        stride (tuple|list, optional): The stride of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        crop_size (tuple|list, optional):  The crop size of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        num_workers (int, optional): Num workers for data loader. Default: 0.

    Returns:
        float: The mIoU of validation datasets.
        float: The accuracy of validation datasets.
    """
    if device == None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    val_loader = DataLoader(dataset=eval_dataset, batch_size=1, shuffle=True, num_workers=num_workers)

    total_iters = len(val_loader)
    intersect_area_all = 0
    pred_area_all = 0
    label_area_all = 0

    logger.info("Start evaluating (total_samples={}, total_iters={})...".format(
        len(eval_dataset), total_iters))
    progbar_val = progbar.Progbar(target=total_iters, verbose=1)
    timer = Timer()
    with torch.no_grad():
        for iter, sample in enumerate(val_loader):
            image, label = sample['image'], sample['label']
            image, label = Variable(image.to(device)), Variable(label.to(device))

            reader_cost = timer.elapsed_time()

            ori_shape = label.shape[-2:]
            if aug_eval:
                pred = infer.aug_inference(
                    model,
                    image,
                    ori_shape=ori_shape,
                    transforms=eval_dataset.transform,
                    scales=scales,
                    flip_horizontal=flip_horizontal,
                    flip_vertical=flip_vertical,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
            else:
                pred = infer.inference(
                    model,
                    image,
                    ori_shape=ori_shape,
                    transforms=eval_dataset.transform,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)

            intersect_area, pred_area, label_area = metrics.calculate_area(
                pred,
                label,
                eval_dataset.num_classes,
                ignore_index=eval_dataset.ignore_index)

            
            intersect_area_all = intersect_area_all + intersect_area
            pred_area_all = pred_area_all + pred_area
            label_area_all = label_area_all + label_area
            batch_cost = timer.elapsed_time()
            timer.restart()

            progbar_val.update(iter + 1, [('batch_cost', batch_cost),
                                            ('reader cost', reader_cost)])

    class_iou, miou = metrics.mean_iou(intersect_area_all, pred_area_all,
                                       label_area_all)
    class_dice, dice = metrics.dice(intersect_area_all, pred_area_all,
                                       label_area_all)
    class_acc, acc = metrics.accuracy(intersect_area_all, pred_area_all)
    kappa = metrics.kappa(intersect_area_all, pred_area_all, label_area_all)

    logger.info("[EVAL] #Images={} mIoU={:.4f} dice={:.4f} Acc={:.4f} Kappa={:.4f} ".format(
        len(eval_dataset), miou, dice, acc, kappa))
    logger.info("[EVAL] Class IoU: \n" + str(np.round(class_iou, 4)))
    logger.info("[EVAL] Class dice: \n" + str(np.round(class_dice, 4)))
    logger.info("[EVAL] Class Acc: \n" + str(np.round(class_acc, 4)))
    return miou, dice, acc
