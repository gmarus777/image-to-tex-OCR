import tarfile
import glob
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import os
import re
import cv2




class Data_Server:
    def __init__(self,
                 datamodule = None,
                 image_transform: str = 'to_tensor',
                 labels_transform='BPE',
                 BPE_set_vocab_size = 8000,
                 train_val_fraction = 0.8,
                 max_label_length = 128,
                 max_number_to_render = 150,
                 image_height = 64,
                 image_width = 512,
                 augment_images = False,
                 number_tex_formulas_to_generate = 150,
                 generate_tex_formulas = True,
                 generate_svg_images_from_tex = True,
                 generate_png_from_svg = True,
                 download_tex_dataset = True,
                 number_png_images_to_use_in_dataset = 120
                 ):


        # Non-tokenized dataframe
        self.raw_dataframe = self.get_statistics()



