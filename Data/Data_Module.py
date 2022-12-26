from pathlib import Path
import argparse
from typing import Dict, Tuple, Collection, Union, Optional
import torch
import json
from torch.utils.data import ConcatDataset, DataLoader
import pytorch_lightning as pl


'''


'''

class Data_Module(pl.LightningDataModule):

    def __init__(self,
                 stage='fit',

                 max_label_length=128,
                 number_png_images_to_use_in_dataset=120,
                 labels_transform='default',
                 image_transform_name='torchvision',  # or 'alb'

                 load_tokenizer = False,
                 train_test_fraction = .01,
                 train_val_fraction=0.8,

                 image_height=64,
                 image_width=512,
                 augment_images=False,

                 batch_size=64,
                 num_workers=10,
                 data_on_gpu=False,

                 max_number_to_render=150,  # placeholder for now needs implementation
                 ):


        '''

        :param stage:
        :param max_label_length:
        :param number_png_images_to_use_in_dataset:
        :param labels_transform:
        :param image_transform_name:
        :param load_tokenizer:
        :param train_test_fraction:
        :param train_val_fraction:
        :param image_height:
        :param image_width:
        :param augment_images:
        :param batch_size:
        :param num_workers:
        :param data_on_gpu:
        :param max_number_to_render:
        '''

        super().__init__()

        # Various input parameters
        self.stage = stage

        self.max_output_label_length = max_label_length
        self.number_png_images_to_use_in_dataset = number_png_images_to_use_in_dataset
        self.labels_transform = labels_transform

        self.load_tokenizer = load_tokenizer


        self.image_transform_name = image_transform_name
        self.image_transform_alb = train_transform
        self.image_transform_test = test_transform


        self.image_height = image_height
        self.image_width = image_width


        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_fraction = train_val_fraction
        self.max_number_to_render = max_number_to_render
        self.on_gpu = data_on_gpu
        self.shuffle_train = True

        # Data Loaders will load model-feeding data here
        self.data_train: Union[BaseDataset, ConcatDataset]
        self.data_val: Union[BaseDataset, ConcatDataset]
        self.data_test: Union[BaseDataset, ConcatDataset]





