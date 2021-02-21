#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
from torch import nn
import argparse
import warnings
import models
from dataset.cifar10_loader import CIFAR10_Loader
from tqdm import tqdm
import logging

class Evaluate_Utils(object):
    def __init__(self, args, model_name, checkpoint_file):
        self.args = args
        self.model_name = model_name
        self.checkpoint_file = checkpoint_file
        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1

        # Load the datasets
        cifar10 = CIFAR10_Loader(root=args.data_dir)
        cifar10_dataset = cifar10.get_dataset(dataset_type='test')
        self.dataloaders = torch.utils.data.DataLoader(cifar10_dataset, batch_size=args.batch_size,
                                                       num_workers=args.num_workers,
                                                       pin_memory=(True if self.device == 'cuda' else False))
        self.dataloaders = tqdm(self.dataloaders)
        # Define the model
        self.model = getattr(models, self.model_name)(in_channel=cifar10.inputchannel, out_channel=cifar10.num_classes)
        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)


        # Load the checkpoint
        self.checkpoint = torch.load(self.checkpoint_file)
        self.model.load_state_dict(self.checkpoint)

        # Invert the model and define the loss
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def evaluate(self):
        args = self.args
        epoch_acc = 0
        epoch_loss = 0.0
        batch_count = 0
        
        self.model.eval()
        for batch_idx, (inputs, labels) in enumerate(self.dataloaders):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(inputs)
            loss = self.criterion(input=logits, target=labels)
            pred = logits.argmax(dim=1)
            correct = torch.eq(pred, labels).float().sum().item()
            loss_temp = loss.item() * inputs.size(0)
            epoch_loss += loss_temp
            epoch_acc += correct
            batch_count += inputs.size(0)
        
        # Print the train and val information via each epoch
        epoch_loss = epoch_loss / batch_count
        epoch_acc = epoch_acc / batch_count

        logging.info("test acc {:.4f}, loss {:.4f}, with model parameters loaded from {}".format(epoch_acc, epoch_loss, self.checkpoint_file))

