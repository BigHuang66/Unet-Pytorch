# 语义分割实验平台

基于pytorch的语义分割实验平台，包括 train、evaluate、predict 等功能

### 1、数据准备

数据存储在工程目录data下，并以命名xxx。子文件夹下包含images、labels文件夹，各自为图片与标注文件；数据划分使用额外的文件方式标注。

```
        data/XXX/
        |
        |--images
        |  |--xxx.jpg
        |  |--...
        |
        |--labels
        |  |--xxx.png
        |  |--...
        |
        |--filelist.josn
        |--train.josn
        |--val.josn
```

### 2、Dataloader 与 数据增强

​		继承 torch.utils.data 中Dataset、Dataloader类改写即可，Dataset定义 见 `utils/datasets.py`

​		数据增强部分为增加多样性，选择 albu 的数据增强工具箱：https://github.com/albumentations-team/albumentations

### 3、快速开始

#### 环境安装

```shell
pip install -r requirement.txt
```

#### 训练 ------ 模型训练与参数保存

​        修改`main.py`中相关数据路径与超参数，终端执行`python main.py --command train`即可实现训练，也可在parser中command修改 default="train"，可实现IDE执行训练过程

#### 验证 ------ 验证某参数下模型性能

​		终端执行`python main.py --command eval`，也可在parser中command修改 default="eval"，可实现IDE执行验证过程

​		**注意：默认使用mAP最高的模型参数进行验证测试，即存储 best_model 下，自定义需修改**

#### 预测 ------ 使用模型进行推理，预测实际结果

​        终端执行`python main.py --command predict`即可实现训练，也可在parser中command修改 default="predict"，可实现IDE执行训练过程

​		**注意：默认使用mAP最高的模型参数进行验证测试，即存储 best_model 下，自定义需修改**

### 4、自定义

​		暂未给出自定义模型、Loss等案例，可模仿标准案例修改

### 5、待添加

- [ ] 滑动窗口预测
- [x] 多尺度训练
- [x] 测试时间增强
- [x] 导入训练
- [ ] 添加常见使用的模型

### 6、引用

本项目只限学习使用，很多功能与实现参考在百度开源语义分割模型 paddleseg，非常感谢！

```
@misc{liu2021paddleseg,
      title={PaddleSeg: A High-Efficient Development Toolkit for Image Segmentation}, 
      author={Yi Liu and Lutao Chu and Guowei Chen and Zewu Wu and Zeyu Chen and Baohua Lai and Yuying Hao},
      year={2021},
      eprint={2101.06175},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{paddleseg2019,
    title={PaddleSeg, End-to-end image segmentation kit based on PaddlePaddle},
    author={PaddlePaddle Authors},
    howpublished = {\url{https://github.com/PaddlePaddle/PaddleSeg}},
    year={2019}
}
```