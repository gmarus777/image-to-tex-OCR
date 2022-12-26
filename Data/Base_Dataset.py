import torch
import os
import numpy as np
import cv2
from Data.configs import PrintedLatexDataConfig
import json
import tarfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
from urllib.request import urlretrieve
from PIL import Image
from torch.utils.data import Dataset


class Base_Dataset(Dataset):

    def __init__(self,
                 data_module,
                 ):

        self.datamodule = data_module
        self.tokenizer = data_module.tokenizer
        self.dataframe = data_module.df

        self.image_filenames = self.dataframe['image_name'].tolist()
        self.labels = self.dataframe['latex_tokenized'].tolist()

        self.image_transform_name = data_module.image_transform_name
        self.image_transform_alb = data_module.image_transform_alb
        self.image_transform_test = data_module.image_transform_test



        self.stage = data_module.stage


        self.labels_transform_function = data_module.labels_transform_function



        if len(self.image_filenames) != len(self.labels):
            raise ValueError("Images and Labels must be of equal lengths")
        super(BaseDataset, self).__init__()




    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index: int):
        """
        returns datum and it's target, after processing by transforms
        :param index:
        :return:
        """
        image_filename = self.image_filenames[index]
        formula = self.labels[index]

        # print('datum', datum) # this is  image
        # print('target', target) # this is tokenized label

        oldcwd = os.getcwd()
        os.chdir(PrintedLatexDataConfig.PROCESSED_DATA_FOLDER)  # Data/processed_data

        # image = pil_loader('generated_png_images/' + image_filename, mode="L")

        image = ImageProcessor.read_image_pil('generated_png_images/' + image_filename, grayscale=True)







        if self.stage.lower() =="fit" and self.image_transform_name.lower() == 'alb':
           #image =  self.image_transform_train(image)
           image = self.image_transform_alb(image=np.array(image))['image']

           formula = self.labels_transform_function(formula)

        if self.stage.lower() =="fit" and self.image_transform_name.lower() == 'torchvision':
           #image =  self.image_transform_train(image)
           image = self.image_transform_torch(image)

           formula = self.labels_transform_function(formula)





        if self.stage == 'test':
            image =  self.image_transform_train(image)
            formula = self.labels_transform_function(formula)

        os.chdir(oldcwd)

        return image, formula








def pil_loader(fp: Path, mode: str) -> Image.Image:
    with open(fp, "rb") as f:
        img = Image.open(f)
        return img.convert(mode)


def split_dataset(base_dataset: BaseDataset, fraction: float):
    """
    Split input base_dataset into 2 base datasets, the first of size fraction * size of the base_dataset and the
    other of size (1 - fraction) * size of the base_dataset.
    """
    split_a_size = int(fraction * len(base_dataset))
    split_b_size = len(base_dataset) - split_a_size



    return torch.utils.data.random_split( base_dataset, [split_a_size, split_b_size])


def get_data_server_class(data_server_class_name):
    return _import_class(f"DataPipes.dataServers.{data_server_class_name}")