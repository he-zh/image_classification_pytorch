# image_classification_pytorch

This repository contains pytorch implementation of some of the convolutional networks referred by the [Dive Into Deep Learning](https://d2l.ai/) book 
([中文版:动手学深度学习](http://zh.d2l.ai/)). 

## Requirements
- python 3.6
- pytorch 1.7.0
- torchvision 0.8.1
- numpy
- tqdm

## Usage

### dataset

- Cifar10 [download](https://www.cs.toronto.edu/~kriz/cifar.html)

Instead of using cifar10 dataset from torchvision, I write code for collecting data from raw files in dataset folder.

### train

train the model using train.py

```
$ python train.py --data_dir=<cifar10-directory> --checkpoint_dir=<where-to-save-checkpoint> --model_name=ResNet18
```

Instead of invoking model from  torchvision.models, I write codes to implement several convolutional neural networks in models folders.

The supported model_name args are:
```
LeNet
VGG11
VGG13
VGG16
VGG19
ResNet18
ResNet34
ResNet50
ResNet101
ResNet152
DenseNet121
DenseNet161
DenseNet169
DenseNet201
```

### test

test the model using evaluate.py

```
$ python evaluate.py --data_dir=<cifar10-directory> --checkpoint_dir=<path-to-particular-saved-model-dir>
```

## Reference
https://github.com/ShusenTang/Dive-into-DL-PyTorch

https://github.com/yunjey/pytorch-tutorial

https://github.com/pytorch/vision/tree/d5096a7f9944fde2619649f2374d866c86e66c32/torchvision/models
