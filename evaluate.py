#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import os
from utils.evaluate_utils import Evaluate_Utils
import logging
from utils.logger import setlogger
args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    # basic parameters
    parser.add_argument('--data_name', type=str, default='cifar10', help='the name of the data')
    parser.add_argument('--data_dir', type=str, default= '..\..\data\cifar10\cifar-10-batches-py', help='the directory of the data')
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', help='the directory to save the model')
    parser.add_argument('--checkpoint_subdir', type=str, default='LeNet_cifar10_0205-023630', help='the sub-directory to save the model')
    parser.add_argument('--batch_size', type=int, default=64, help='batchsize of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()

    # get model name  
    model_name = args.checkpoint_subdir.split("_")[0]

    # get the checkpoint file of best model
    path = os.path.join(args.checkpoint_dir, args.checkpoint_subdir)
    f_list = os.listdir(path)
    files = [i for i in f_list if '.pth' in i]
    files.sort(key=lambda x:float(x.split('-')[1]), reverse=True)
    checkpoint_file = os.path.join(path,files[0])

    # set the test logger
    setlogger(os.path.join(path, 'test.log'))

    evaluater = Evaluate_Utils(args, model_name, checkpoint_file)
    evaluater.evaluate()