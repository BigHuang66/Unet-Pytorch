
import os

import cv2
import numpy as np
import torch

from utils import visualize
from tools import infer
from utils import logger, progbar


def mkdir(path):
    sub_dir = os.path.dirname(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)


def predict(model,
            model_path,
            transforms,
            image_list,
            image_dir=None,
            save_dir='output',
            device=None,
            aug_pred=False,
            scales=1.0,
            flip_horizontal=True,
            flip_vertical=False,
            is_slide=False,
            stride=None,
            crop_size=None):
    """
    predict and visualize the image_list.

    Args:
        model (nn.Layer): Used to predict for input image.
        model_path (str): The path of pretrained model.
        transforms (transform.Compose): Preprocess for input image.
        image_list (list): A list of image path to be predicted.
        image_dir (str, optional): The root directory of the images predicted. Default: None.
        save_dir (str, optional): The directory to save the visualized results. Default: 'output'.
        aug_pred (bool, optional): Whether to use mulit-scales and flip augment for predition. Default: False.
        scales (list|float, optional): Scales for augment. It is valid when `aug_pred` is True. Default: 1.0.
        flip_horizontal (bool, optional): Whether to use flip horizontally augment. It is valid when `aug_pred` is True. Default: True.
        flip_vertical (bool, optional): Whether to use flip vertically augment. It is valid when `aug_pred` is True. Default: False.
        is_slide (bool, optional): Whether to predict by sliding window. Default: False.
        stride (tuple|list, optional): The stride of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        crop_size (tuple|list, optional):  The crop size of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.

    """
    para_state_dict = torch.load(model_path)
    model.load_state_dict(para_state_dict)
    if device == None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    added_saved_dir = os.path.join(save_dir, 'added_prediction')
    pred_saved_dir = os.path.join(save_dir, 'pseudo_color_prediction')

    logger.info("Start to predict...")
    progbar_pred = progbar.Progbar(target=len(image_list), verbose=1)
    with torch.no_grad():
        for i, im_path in enumerate(image_list):
            im = cv2.imread(os.path.join(image_dir, im_path))
            ori_shape = im.shape[:2]
            transformed = transforms(image=im, mask=None)
            im = transformed['image']
            im = im[np.newaxis, ...].to(device)

            if aug_pred:
                pred = infer.aug_inference(
                    model,
                    im,
                    ori_shape=ori_shape,
                    transforms=transforms.transforms,
                    scales=scales,
                    flip_horizontal=flip_horizontal,
                    flip_vertical=flip_vertical,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
            else:
                pred = infer.inference(
                    model,
                    im,
                    ori_shape=ori_shape,
                    transforms=transforms.transforms,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
            pred = torch.squeeze(pred)
            pred = pred.cpu().numpy().astype('uint8')

            # get the saved name
            if image_dir is not None:
                im_file = im_path.replace(image_dir, '')
            else:
                im_file = os.path.basename(im_path)
            if im_file[0] == '/':
                im_file = im_file[1:]

            # save added image
            added_image = visualize.visualize(os.path.join(image_dir, im_path), pred, weight=0.6)
            added_image_path = os.path.join(added_saved_dir, im_file)
            mkdir(added_image_path)
            cv2.imwrite(added_image_path, added_image)

            # save pseudo color prediction
            pred_mask = visualize.get_pseudo_color_map(pred)
            pred_saved_path = os.path.join(pred_saved_dir,
                                           im_file.rsplit(".")[0] + ".png")
            mkdir(pred_saved_path)
            pred_mask.save(pred_saved_path)

            # pred_im = visualize(im_path, pred, weight=0.0)
            # pred_saved_path = os.path.join(pred_saved_dir, im_file)
            # mkdir(pred_saved_path)
            # cv2.imwrite(pred_saved_path, pred_im)

            progbar_pred.update(i + 1)
