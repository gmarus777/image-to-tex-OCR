from pathlib import Path
import argparse
from typing import Dict, Tuple, Collection, Union, Optional
import torch
import json
import random
from torch.utils.data import ConcatDataset, DataLoader
import pytorch_lightning as pl
from Data.label_transforms import Label_Transforms
from Data.image_transforms import Image_Transforms
from Data.Data_Server import Data_Server
from Data.Base_Dataset import Base_Dataset, split_dataset
from Data.vocabulary_utils import load_dic, invert_vocabulary
from torchvision import transforms
import torch.nn.functional as F







'''

Start, End, Pad tokens are set in vocabulary_utils.py

IMAGE_HEIGHT and IMAGE_WIDTH are set in image_transforms.py (Note this will just resize the generated images)

'''

class Data_Module(pl.LightningDataModule):

    def __init__(self,
                 stage='fit',
                 path_to_formulas = None,
                 path_to_image_names = None,


                 set_max_label_length=128,
                 number_png_images_to_use_in_dataset=300*1000,
                 labels_transform='default',
                 image_transform_name='alb',  # or 'alb'
                max_width = 700,
                 image_padding = False,
                 load_vocabulary = False,
                 vocabulary_path = None,

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
        self.path_to_formulas = path_to_formulas,
        self.path_to_image_names = path_to_image_names,

        self.set_max_label_length = set_max_label_length
        self.number_png_images_to_use_in_dataset = number_png_images_to_use_in_dataset
        self.labels_transform = labels_transform

        self.load_vocabulary = load_vocabulary
        self.vocabulary_path = vocabulary_path


        self.image_transform_name = image_transform_name

        self.image_padding = image_padding

        self.image_transform_alb = Image_Transforms.train_transform_with_padding
        self.image_transform_alb_small = Image_Transforms.train_transform_with_padding_small
        self.image_transform_alb_xs = Image_Transforms.train_transform_with_padding_xs



        self.image_transform_test = Image_Transforms.test_transform_with_padding
        self.image_transform_test_small = Image_Transforms.test_transform_with_padding_small
        self.image_transform_test_medium = Image_Transforms.test_transform_with_padding_medium
        self.image_transform_test_xl = Image_Transforms.test_transform_with_padding_xl
        self.image_transform_test_xs =Image_Transforms.test_transform_with_padding_xs




        self.max_width = max_width

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
    def prepare_data(self, *args, **kwargs):
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
        self.vocabulary = load_dic(self.vocabulary_path)
        self.vocab_size = len(self.vocabulary)
        self.inverse_vocabulary = invert_vocabulary(self.vocabulary)
        self.max_label_length = int(self.set_max_label_length) + int(2)
        self.tokenizer = Label_Transforms(vocabulary = self.vocabulary,
                                          labels_transform_name = self.labels_transform,
                                          max_label_length = int(self.max_label_length))

        # funciton to turn strings into labels via a tokenizer
        self.labels_transform_function = self.tokenizer.convert_strings_to_labels





    def setup(self, stage = None):

        if stage == "fit" or stage is None:
            data_trainval = Base_Dataset(data_module = self)
            self.data_train, self.data_val = split_dataset(base_dataset = data_trainval, fraction = self.train_val_fraction)
            print('Train/Val Data is ready for Model loading.')

        if stage == 'test':
            self.data_test = [0]



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
            pin_memory=self.on_gpu,
            collate_fn=collate_function,
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
            pin_memory=self.on_gpu,
            collate_fn=collate_function
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
            pin_memory=self.on_gpu,
            collate_fn=collate_function,
        )


def collate_function_old( batch):
    images, formulas = zip(*batch)
    B = len(images)
    max_H = max(image.shape[1] for image in images)
    max_W = max(image.shape[2] for image in images)
    max_length = max(len(formula) for formula in formulas)
    padded_images = torch.zeros((B, 1, max_H, max_W))
    batched_indices = torch.zeros((B, max_length ), dtype=torch.long)
    for i in range(B):
        H, W = images[i].shape[1], images[i].shape[2]
        y, x = random.randint(0, max_H - H), random.randint(0, max_W - W)
        padded_images[i, :, y : y + H, x : x + W] = images[i]
        indices = formulas[i]
        #batched_indices[i, : len(indices)] = torch.tensor(indices, dtype=torch.long)
        batched_indices[i, : len(indices)] = indices.clone().detach()
    return padded_images, batched_indices


def collate_function(batch):
    # Get the maximum height of images in the batch
    images, formulas = zip(*batch)
    B = len(images)
    max_H = max(image.shape[1] for image in images)
    max_W = max(image.shape[2] for image in images)


    # Pad images to the maximum height using zero-padding
    padded_images = []
    labels = formulas

    for i in range(B):
        H, W = images[i].shape[1], images[i].shape[2]
        padding_width = max_W-W
        padding_height = max_H - H
        #padding = transforms.Pad((0, padding_height, 0, padding_width), fill=1)
        padded_image = F.pad(images[i], (0, max_W - W, 0, max_H - H), value=1)
        #padded_image = padding(images[i])
        padded_images.append(padded_images)

    # Stack the padded images and labels into a batch tensor
    images = torch.stack(images)
    labels =torch.stack(labels)
    return images, labels