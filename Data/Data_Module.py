from pathlib import Path
import argparse
from typing import Dict, Tuple, Collection, Union, Optional
import torch
import json
from torch.utils.data import ConcatDataset, DataLoader
import pytorch_lightning as pl
from Data.label_transforms import Label_Transforms
from Data.image_transforms import train_transform, test_transform
from Data.Data_Server import Data_Server



'''

Start, End, Pad tokens are set in vocabulary_utils.py

'''

class Data_Module(pl.LightningDataModule):

    def __init__(self,
                 stage='fit',

                 set_max_label_length=256,
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

        self.set_max_label_length = set_max_label_length
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



    # Uses images 'Data/generated_png_images/', formulas 'Data/final_png_formulas.txt'
    # and image filenames 'Data/corresponding_png_images.txt'
    # to generate a pandas tokenized dataframe

    def prepare_dataframe(self, *args, **kwargs):
        self.data_server = Data_Server(data_module=self)
        self.df = self.data_server.tokenized_dataframe
        self.vocabulary = self.data_server.vocabulary
        self.inverse_vocabulary = self.data_server.inverse_vocabulary
        self.max_label_length = self.data_server.max_label_length





        # self.dataframe = self.data_server.tokenized_dataframe
        # self.max_label_length = self.data_server.max_label_length


