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
from Data.Base_Dataset import Base_Dataset, split_dataset
from Data.vocabulary_utils import load_dic, invert_vocabulary


# Use the following path when Loading a Vocabulary
VOCABULARY_PATH = 'lightning_logs/256_character_1.json'






'''

Start, End, Pad tokens are set in vocabulary_utils.py

IMAGE_HEIGHT and IMAGE_WIDTH are set in image_transforms.py (Note this will just resize the generated images)

'''

class Data_Module(pl.LightningDataModule):

    def __init__(self,
                 stage='fit',

                 set_max_label_length=256,
                 number_png_images_to_use_in_dataset=200*1000,
                 labels_transform='default',
                 image_transform_name='alb',  # or 'alb'

                 load_vocabulary = False,
                 train_val_fraction=0.9,


                 batch_size=64,
                 num_workers=10,
                 data_on_gpu=False,

                 ):


        '''

        :param stage:
        :param max_label_length:
        :param number_png_images_to_use_in_dataset:
        :param labels_transform:
        :param image_transform_name:
        :param load_vocabulary:
        :param train_test_fraction:
        :param train_val_fraction:
        :param augment_images:
        :param batch_size:
        :param num_workers:
        :param data_on_gpu:

        '''

        super().__init__()

        # Various input parameters
        self.stage = stage

        self.set_max_label_length = set_max_label_length
        self.number_png_images_to_use_in_dataset = number_png_images_to_use_in_dataset
        self.labels_transform = labels_transform

        self.load_vocabulary = load_vocabulary

        self.image_transform_name = image_transform_name
        self.image_transform_alb = train_transform
        self.image_transform_test = test_transform

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_fraction = train_val_fraction
        self.on_gpu = data_on_gpu
        self.shuffle_train = True


        if load_vocabulary == True:
            self.load_tokenizer()


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
        self.vocab_size = len(self.vocabulary)
        self.tokenizer = Label_Transforms(vocabulary=self.vocabulary,
                                          labels_transform_name=self.labels_transform,
                                          max_label_length=self.max_label_length)

        # funciton to turn strings into labels via a tokenizer
        self.labels_transform_function = self.tokenizer.convert_strings_to_labels




    def load_tokenizer(self, *args, **kwargs):
        self.vocabulary = load_dic(VOCABULARY_PATH)
        self.vocab_size = len(self.vocabulary)
        self.inverse_vocabulary = invert_vocabulary(self.vocabulary)
        self.tokenizer = Label_Transforms(vocabulary = self.vocabulary,
                                          labels_transform_name = self.labels_transform,
                                          max_label_length = int(self.set_max_label_length)+int(2))

        # funciton to turn strings into labels via a tokenizer
        self.labels_transform_function = self.tokenizer.convert_strings_to_labels




    def setup(self):
        stage = self.stage

        if stage == "fit" or stage is None:
            data_trainval = Base_Dataset(data_module = self)
            self.data_train, self.data_val = split_dataset(base_dataset = data_trainval, fraction = self.train_val_fraction)
            print('Train/Val Data is ready for Model loading.')

        if stage == 'test':
            self.data_test = self.data_server.serve_test_dataset()



    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        """
        construct a dataloader for training data
        data is shuffled !
        :param args:
        :param kwargs:
        :return:
        """
        return DataLoader(
            self.data_train,
            shuffle=self.shuffle_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu
        )

    def val_dataloader(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        return DataLoader(
            self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu
        )

    def test_dataloader(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        return DataLoader(
            self.data_test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu
        )




