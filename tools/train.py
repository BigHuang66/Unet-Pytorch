# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts
from torch import optim
from torch.autograd import Variable
import torch.nn as nn
from utils import Timer, calculate_eta, logger
from tools.val import evaluate
from model.losses import DiceLoss, CrossEntropyLoss, FocalLoss, LovaszSoftmax


# def check_logits_losses(logits_list, losses):
#     len_logits = len(logits_list)
#     len_losses = len(losses['types'])
#     if len_logits != len_losses:
#         raise RuntimeError(
#             'The length of logits_list should equal to the types of loss config: {} != {}.'
#             .format(len_logits, len_losses))


# def loss_computation(logits_list, labels, losses, edges=None):
#     check_logits_losses(logits_list, losses)
#     loss = 0
#     for i in range(len(logits_list)):
#         logits = logits_list[i]
#         loss_i = losses['types'][i]
#         # Whether to use edges as labels According to loss type .
#         if loss_i.__class__.__name__ in ('BCELoss', ) and loss_i.edge_label:
#             loss += losses['coef'][i] * loss_i(logits, edges)
#         else:
#             loss += losses['coef'][i] * loss_i(logits, labels)
#     return loss


def train(params,
          model,
          train_dataset,
          val_dataset=None,
          device=None,
          optimizer=None,
          resume_model=None,
          use_vdl=False,
          losses=None):
    """
    Launch training.

    Args:
        model (nn.Layer): A sementic segmentation model.
        train_dataset (paddle.io.Dataset): Used to read and process training datasets.
        val_dataset (paddle.io.Dataset, optional): Used to read and process validation datasets.
        optimizer (paddle.optimizer.Optimizer): The optimizer.
        save_dir (str, optional): The directory for saving the model snapshot. Default: 'output'.
        iters (int, optional): How may iters to train the model. Defualt: 10000.
        batch_size (int, optional): Mini batch size of one gpu or cpu. Default: 2.
        resume_model (str, optional): The path of resume model.
        save_interval (int, optional): How many iters to save a model snapshot once during training. Default: 1000.
        log_iters (int, optional): Display logging information at every log_iters. Default: 10.
        num_workers (int, optional): Num workers for data loader. Default: 0.
        use_vdl (bool, optional): Whether to record the data to VisualDL during training. Default: False.
        losses (dict): A dict including 'types' and 'coef'. The length of coef should equal to 1 or len(losses['types']).
            The 'types' item is a list of object of paddleseg.models.losses while the 'coef' item is a list of the relevant coefficient.
    """
   #TODO 多卡模式

    max_iters      = params['max_iters']
    batch_size     = params['batch_size']
    lr             = params['lr']
    gamma          = params['gamma']
    step_size      = params['step_size']
    momentum       = params['momentum']
    weight_decay   = params['weight_decay']
    log_iters     = params['log_iters']
    save_iters     = params['save_iters']
    ckpt_dir       = params['ckpt_dir']
    num_workers    = params['num_workers']
    

    start_iter = 0

    if not os.path.isdir(ckpt_dir):
        if os.path.exists(ckpt_dir):
            os.remove(ckpt_dir)
        os.makedirs(ckpt_dir)
        
    if use_vdl:
        from visualdl import LogWriter
        log_dir = os.path.join(ckpt_dir, 'logs')
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        log_writer = LogWriter(log_dir)
    if device == None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
  
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=150*5, T_mult=2)
    
    diceloss = DiceLoss().to(device)
    crossentropy = CrossEntropyLoss().to(device)

    if resume_model != None:
        if os.path.exists(resume_model):
            logger.info('Loading pretrained model from {}'.format(resume_model))
            checkpoint = torch.load(resume_model)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optim_state_dict'])
            criterion.load_state_dict(checkpoint['criterion_state_dict'])
            start_iter = checkpoint['iter']
        else:
            raise ValueError(
                'The pretrained model directory is not Found: {}'.format(
                    pretrained_model))


    timer = Timer()
    avg_loss = 0.0
    iters_per_epoch = len(train_loader)
    best_mean_iou = -1.0
    best_model_iter = -1
    train_reader_cost = 0.0
    train_batch_cost = 0.0
    timer.start()

    iter = start_iter
    while iter < max_iters:
        for samples in train_loader:
            iter += 1
            if iter > max_iters:
                break
            train_reader_cost += timer.elapsed_time()
            images, labels = samples['image'], samples['label']
            images, labels = Variable(images.to(device)), Variable(labels.to(device))
            optimizer.zero_grad()
            pred = model(images)
            loss = diceloss(pred, labels.long()) + crossentropy(pred, labels.long())
            loss.backward()
            optimizer.step()
           
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            scheduler.step()

            avg_loss += loss.item()
            train_batch_cost += timer.elapsed_time()

            if (iter) % log_iters == 0:
                avg_loss /= log_iters
                avg_train_reader_cost = train_reader_cost / log_iters
                avg_train_batch_cost = train_batch_cost / log_iters
                train_reader_cost = 0.0
                train_batch_cost = 0.0
                remain_iters = max_iters - iter
                eta = calculate_eta(remain_iters, avg_train_batch_cost)
                logger.info(
                    "[TRAIN] epoch={}, iter={}/{}, loss={:.4f}, lr={:.6f}, batch_cost={:.4f}, reader_cost={:.4f} | ETA {}"
                    .format((iter - 1) // iters_per_epoch + 1, iter, max_iters,
                            avg_loss, lr, avg_train_batch_cost,
                            avg_train_reader_cost, eta))
                if use_vdl:
                    log_writer.add_scalar('Train/loss', avg_loss, iter)
                    log_writer.add_scalar('Train/lr', lr, iter)
                    log_writer.add_scalar('Train/batch_cost',
                                          avg_train_batch_cost, iter)
                    log_writer.add_scalar('Train/reader_cost',
                                          avg_train_reader_cost, iter)
                avg_loss = 0.0

            if (iter % save_iters == 0 or iter == max_iters) and (val_dataset is not None):
                num_workers = 1 if num_workers > 0 else 0
                mean_iou, dice, acc = evaluate(
                    model, val_dataset, device=device, num_workers=num_workers)
                model.train()

            if iter % save_iters == 0 or iter == max_iters:
                current_save_dir = os.path.join(ckpt_dir,
                                                "iter_{}".format(iter))
                if not os.path.isdir(current_save_dir):
                    os.makedirs(current_save_dir)
                checkpoint_dict = {'iter': iter,
                                    'model_state_dict': model.state_dict(),
                                    'optim_state_dict': optimizer.state_dict()}
                                    # 'criterion_state_dict': criterion.state_dict()}
                torch.save(checkpoint_dict,
                            os.path.join(current_save_dir, 'model.pth'))

                if val_dataset is not None:
                    if mean_iou > best_mean_iou:
                        best_mean_iou = mean_iou
                        best_model_iter = iter
                        best_model_dir = os.path.join(ckpt_dir, "best_model")
                        if not os.path.isdir(best_model_dir):
                            os.makedirs(best_model_dir)
                        torch.save(
                            model.state_dict(),
                            os.path.join(best_model_dir, 'model.pth'))
                    logger.info(
                        '[EVAL] The model with the best validation mIoU ({:.4f}) was saved at iter {}.'
                        .format(best_mean_iou, best_model_iter))

                    if use_vdl:
                        log_writer.add_scalar('Evaluate/mIoU', mean_iou, iter)
                        log_writer.add_scalar('Evaluate/dice', dice, iter)
                        log_writer.add_scalar('Evaluate/Acc', acc, iter)
            timer.restart()

    # Sleep for half a second to let dataloader release resources.
    time.sleep(0.5)
    if use_vdl:
        log_writer.close()
