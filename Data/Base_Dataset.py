import torch
import os
import numpy as np
import cv2
from Data.configs import PrintedLatexDataConfig
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset, ConcatDataset
import PIL
import smart_open
from PIL import Image
import torch
from Data.image_transforms import Image_Transforms
import torch.nn.functional as F
import random
import imutils

MAX_RATIO = 15
MAX_WIDTH= 1920
MAX_HEIGHT = 128




class Base_Dataset(Dataset):

    def __init__(self,
                 data_module,
                 ):

        self.datamodule = data_module
        self.tokenizer = data_module.tokenizer
        self.dataframe = data_module.df
        self.stage = data_module.stage

        # image filenames and corresponding tex formulas
        self.image_filenames = self.dataframe['image_name'].tolist()
        self.labels = self.dataframe['latex_tokenized'].tolist()

        self.image_transform_name = data_module.image_transform_name
        self.image_transform_alb = data_module.image_transform_alb
        self.image_transform_alb_small = data_module.image_transform_alb_small
        self.image_transform_alb_xs = data_module.image_transform_alb_xs

        self.image_transform_test = data_module.image_transform_test
        self.image_transform_test_small = data_module.image_transform_test_small

        # funciton to turn strings into labels via a tokenizer
        self.labels_transform_function = data_module.labels_transform_function



        if len(self.image_filenames) != len(self.labels):
            raise ValueError("Images and Labels must be of equal lengths")
        super(Base_Dataset, self).__init__()




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

        # Change path to the image folder
        oldcwd = os.getcwd()
        os.chdir(PrintedLatexDataConfig.DATA_BANK_DIRNAME)  # "Data/Data_Bank"
        image = Image.open('generated_png_images/' + image_filename).convert('RGB')
        # image = pil_loader('generated_png_images/' + image_filename, mode="L")
        #image = ImageProcessor.read_image_pil('generated_png_images/' + image_filename, grayscale=True)
        #image = np.asarray(image)
        #h, w, c = image.shape
        #image = cv2.imread('generated_png_images/' + image_filename)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = cv2.bitwise_not(image)

        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #image = cv2.threshold(image, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        #dist = cv2.distanceTransform(image, cv2.DIST_L2, 5)
        #dist = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
        #dist = (dist * 255).astype("uint8")
        #image = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # added inversion
        # image = PIL.ImageOps.invert(image)
        #h, w = image.shape
        #w,h = image.size

        #ratio = int((w / h))
        #if ratio == 0:
            #ratio = 1
        #if ratio>MAX_RATIO:
            #ratio =MAX_RATIO
        #new_scale = random.randint(0,2)*random.random()+1
        #factor = random.randint(1,3)
        #w_new = w*factor
        #h_new = h*factor




        # NEW
        #image = cv2.imread('generated_png_images/' + image_filename)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = np.asarray(image)


        #h, w, c = image.shape
        #if w<128 and h<30:
            #image = cv2.resize(image, (0, 0), fx=2+random.random(), fy=2++random.random(), interpolation=cv2.INTER_CUBIC)
        #elif w<300:
            #image = cv2.resize(image, (0, 0), fx=1.5+random.random(), fy=1.5+random.random(),  interpolation=cv2.INTER_CUBIC)
        #elif w<510:
            #image = cv2.resize(image, (0, 0), fx=1+random.random(), fy=1+random.random(),  interpolation=cv2.INTER_CUBIC)

        #image = imutils.resize(image, height=96)
        if self.stage.lower() =="fit":

            image = Image_Transforms.train_transform_with_padding(image=np.array(image))['image'][:1]

            formula = self.labels_transform_function(formula)



        if self.stage == 'test':
            image = Image_Transforms.train_transform_with_padding(image=np.array(image))['image'][:1]
            formula = self.labels_transform_function(formula)

        # try PADDING on the right?
        #image = F.pad(image, (0, MAX_WIDTH - new_w, 0, MAX_HEIGHT - new_h), value=1)
        # change path back
        os.chdir(oldcwd)

        return image, formula








def pil_loader(fp: Path, mode: str) -> Image.Image:
    with open(fp, "rb") as f:
        img = Image.open(f)
        return img.convert(mode)


def split_dataset(base_dataset: Base_Dataset, fraction: float):
    """
    Split input base_dataset into 2 base datasets, the first of size fraction * size of the base_dataset and the
    other of size (1 - fraction) * size of the base_dataset.
    """
    split_a_size = int(fraction * len(base_dataset))
    split_b_size = len(base_dataset) - split_a_size



    return torch.utils.data.random_split( base_dataset, [split_a_size, split_b_size])




# processing images for data generation.
class ImageProcessor:
    @staticmethod
    def process_latex_image(file_name):
        image = ImageProcessor.read_image_pil(file_name)
        image = PIL.ImageOps.grayscale(image)
        image = PIL.ImageOps.invert(image)
        return image

    @staticmethod
    def read_image_pil(image_uri, grayscale=False) -> PIL.Image:
        #print(os.getcwd())
        with smart_open.open(image_uri, "rb") as image_file:
            return ImageProcessor.read_image_pil_file(image_file, grayscale)

    @staticmethod
    def read_image_pil_file(image_file, grayscale=False) -> PIL.Image:
        with Image.open(image_file) as image:
            if grayscale:
                image = image.convert(mode="L")
            else:
                image = image.convert(mode=image.mode)

            return image


