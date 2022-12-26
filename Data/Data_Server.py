import tarfile
import glob
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import os
import re
import cv2
from Data.configs import PrintedLatexDataConfig




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





########## Methods to generate Pandas DataFrame #########

    def get_statistics(self):
        # get dataframe
        formulas_df = _get_dataframe()

        return _get_stats(formulas_df)





####### Helper Functions for Pandas DataFrame generation ###########

def _get_dataframe():
    # take final formula list
    path_to_formulas = PrintedLatexDataConfig.PNG_FINAL_FORMULAS
    formulas_df = readlines_to_df(path_to_formulas, 'formula')

    # get png image names
    image_names_path = PrintedLatexDataConfig.PNG_IMAGES_NAMES_FILE
    image_names_df = readlines_to_df(image_names_path, 'image_name')

    formulas_df['image_name'] = image_names_df

    return formulas_df


# outputs formula length, image height and width.
def _get_stats(datasetDF):
    widths = []
    heights = []
    formula_lens = []

    dataset = datasetDF
    for _, row in datasetDF.iterrows():
        image_name = row.image_name
        # print(image_name)
        im = Image.open(os.path.join(PrintedLatexDataConfig.GENERATED_PNG_DIR_NAME, image_name))
        widths.append(im.size[0])
        heights.append(im.size[1])
        formula_lens.append(len(row.formula))

    # datasetDF = datasetDF.assign(width=widths, height=heights, formula_len=formula_lens)
    dataset['height'] = heights
    dataset['width'] = widths
    dataset['formula_length'] = formula_lens

    return dataset


# converts formulas txt to pandas dataframe
def readlines_to_df(path, colname):
    #   return pd.read_csv(output_file, sep='\t', header=None, names=['formula'], index_col=False, dtype=str, skipinitialspace=True, skip_blank_lines=True)
    rows = []
    n = 0
    with open(path, 'r') as f:
        # print('opened file %s' % path)
        for line in f:
            n += 1
            line = line.strip()  # remove \n
            if len(line) > 0:
                rows.append(line)
    # print('processed %d lines resulting in %d rows' % (n, len(rows)))
    return pd.DataFrame({colname: rows}, dtype=np.str_)


