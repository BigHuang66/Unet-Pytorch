
import os
import cv2
import torch
import argparse
from model import UNet
from utils.datasets import train_transform, val_transform, CustomDataset
from tools import train, evaluate, predict

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--command", default="eval", choices=['train', 'eval', 'predict'],
                        help='train or test a model')
    return parser.parse_args()      

PROJECT = "caries"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


data_dir = "./data/caries"
train_set = 'train.json'
val_set = 'val.json'

train_dataset = CustomDataset(data_dir, image_set_file=train_set, transform=train_transform)
val_dataset = CustomDataset(data_dir, image_set_file=val_set, transform=val_transform)

ckpt_dir = os.path.join('./output', PROJECT)
if not os.path.exists('./output'):
    os.mkdir('./output')
if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)

model = UNet(3, 3).to(device)

param = {}
param['max_iters'] = 15000        # 训练step数
param['batch_size'] = 2        # batch_size
param['lr'] = 1e-2             # 学习率
param['gamma'] = 0.6           # 学习率衰减系数
param['step_size'] = 1000        # 学习率衰减间隔
param['momentum'] = 0.9        # 动量
param['weight_decay'] = 4.0e-5    # 权重衰减
param['ckpt_dir'] = ckpt_dir
param['log_iters'] = 50       # 显示间隔
param['save_iters'] = 500        # 保存间隔
param['num_workers'] = 0

if __name__ == "__main__":
    args = parse_args()

    if args.command == 'train':
        train(param, model, train_dataset, val_dataset, device=device, use_vdl=True)
    elif args.command == 'eval':
        best_miou_model = os.path.join(ckpt_dir, 'best_model', 'model.pth')
        model.load_state_dict(torch.load(best_miou_model))
        mean_iou, dice, acc = evaluate(model, val_dataset, device=device, num_workers=0, aug_eval=True) #, is_slide=True, stride=(64, 64), crop_size=(128, 128))
    elif args.command == 'predict':
        best_miou_model = os.path.join(ckpt_dir, 'best_model', 'model.pdparams')
        predict(model, best_miou_model, transforms=val_dataset.transform, 
                image_list=train_dataset.file_list, image_dir=val_dataset.image_path, 
                save_dir=ckpt_dir, device=device)


