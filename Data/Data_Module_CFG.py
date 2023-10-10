from pathlib import Path
import argparse
from typing import Dict, Tuple, Collection, Union, Optional
import torch
import json
import random
from torch.utils.data import ConcatDataset, DataLoader
import pytorch_lightning as pl
from Data.label_transforms import Label_Transforms

from Data.Data_Server import Data_Server

from Data.Tex_Dataset import Tex_Dataset
from Data.vocabulary_utils import load_dic, invert_vocabulary
from torchvision import transforms
import torch.nn.functional as F
import albumentations as A


MAX_HEIGHT =160
MAX_WIDTH =1280




'''

Start, End, Pad tokens are set in vocabulary_utils.py

IMAGE_HEIGHT and IMAGE_WIDTH are set in image_transforms.py (Note this will just resize the generated images)

'''

class Data_Module_CFG(pl.LightningDataModule):

    def __init__(self,
                cfg,
                 ):



        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg

        self.labels_transform = 'default'

        # Get transforms
        self.train_transform = self.get_transforms('train')
        self.val_transform = self.get_transforms('val')
        self.test_transform = self.get_transforms('test')





        if  self.cfg.load_vocabulary == True:
            self.load_tokenizer()


        # Data Loaders will load model-feeding data here
        self.data_train: Union[Tex_Dataset, ConcatDataset]
        self.data_val: Union[Tex_Dataset, ConcatDataset]
        self.data_test: Union[Tex_Dataset, ConcatDataset]

        # Uses images 'Data/generated_png_images/', formulas 'Data/final_png_formulas.txt'
        # and image filenames 'Data/corresponding_png_images.txt'
        # to generate a pandas tokenized dataframe


        self.data_server = Data_Server(data_module=self)
        self.df = self.data_server.tokenized_dataframe
        # self.vocabulary = self.data_server.vocabulary
        # self.inverse_vocabulary = self.data_server.inverse_vocabulary
        self.max_label_length = self.data_server.max_label_length
        # self.vocab_size = len(self.vocabulary)
        # self.tokenizer = Label_Transforms(vocabulary=self.vocabulary,labels_transform_name=self.labels_transform, max_label_length=self.max_label_length)

        # funciton to turn strings into labels via a tokenizer
        # self.labels_transform_function = self.tokenizer.convert_strings_to_labels



        if self.cfg.stage == "train" or self.cfg.stage is None:


            self.train_indices, self.val_indices = torch.utils.data.random_split(self.df,
                                                                                 [self.cfg.train_val_fraction, 1-self.cfg.train_val_fraction])


            #print(self.val_indices)
            self.data_train = Tex_Dataset(data_module=self ,remove_indices=self.val_indices.indices, stage='train',
                                          image_transform_train=self.train_transform, image_transform_val=None)

            self.data_val = Tex_Dataset(data_module=self ,remove_indices=self.train_indices.indices, stage='val',
                                          image_transform_train=None, image_transform_val=self.val_transform)


            print('Train/Val Data is ready for Model loading.')



            if self.cfg.stage == 'test':
                pass

    def get_transforms(self, stage):
        if stage.lower() == 'train':
            transforms = A.Compose(self.cfg.train_transforms)
        elif stage.lower() == 'val':
            transforms = A.Compose(self.cfg.val_transforms)
        elif stage.lower() == 'test':
            transforms = A.Compose(self.cfg.test_transforms)
        return transforms





    def load_tokenizer(self, *args, **kwargs):
        self.vocabulary = load_dic(self.cfg.vocabulary_path)
        self.vocab_size = len(self.vocabulary)
        self.inverse_vocabulary = invert_vocabulary(self.vocabulary)
        self.max_label_length = int(self.cfg.set_max_label_length) + int(2)
        self.tokenizer = Label_Transforms(vocabulary = self.vocabulary,
                                          labels_transform_name = self.cfg.labels_transform,
                                          max_label_length = int(self.max_label_length))

        # funciton to turn strings into labels via a tokenizer
        self.labels_transform_function = self.tokenizer.convert_strings_to_labels








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
            shuffle=True,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.on_gpu,
            #multiprocessing_context="spawn"
            #multiprocessing_context="fork",
            #collate_fn=self.collate_function,
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
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.on_gpu,
            #multiprocessing_context="spawn",
            #multiprocessing_context="fork",
            #collate_fn=self.collate_function,
            #multiprocessing_context='fork' if torch.backends.mps.is_available() else None,
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
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.on_gpu,
            collate_fn=self.collate_function,
        )


    def collate_function_old(self, batch):
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







    def collate_function(self,batch):
        # Get the maximum height of images in the batch
        images, formulas = zip(*batch)
        B = len(images)
        #max_H = max(image.shape[1] for image in images)
        #max_W = max(image.shape[2] for image in images)
        max_H = MAX_HEIGHT
        max_W = MAX_WIDTH
        #max_L =  max(len(formula) for formula in formulas)


        # Pad images to the maximum height using zero-padding
        padded_images = []
        labels = formulas

        for i in range(B):
            H, W = images[i].shape[1], images[i].shape[2]
            padding_width = max_W-W
            padding_height = max_H - H
            #padding = transforms.Pad((0, padding_height, 0, padding_width), fill=1)
            padded_image = F.pad(images[i], (0, max_W - W, 0, max_H - H), value=0)
            #padded_image = padding(images[i])
            padded_images.append(padded_image)
            #padded_formula =  self.labels_transform_function(string=formulas[i], length=max_L+2)
            #labels.append(padded_formula)

        # Stack the padded images and labels into a batch tensor
        images = torch.stack(padded_images)
        labels =torch.stack(formulas)
        return images, labels

def split_dataset(base_dataset: Tex_Dataset, fraction: float):
    """
    Split input base_dataset into 2 base datasets, the first of size fraction * size of the base_dataset and the
    other of size (1 - fraction) * size of the base_dataset.
    """
    split_a_size = int(fraction * len(base_dataset))
    split_b_size = len(base_dataset) - split_a_size



    return torch.utils.data.random_split( base_dataset, [split_a_size, split_b_size])