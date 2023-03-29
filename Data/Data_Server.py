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
from Data.vocabulary_utils import create_vocabulary_dictionary_from_dataframe, make_vocabulary, invert_vocabulary
import imutils




class Data_Server:
    def __init__(self,
                 data_module = None,
                 ):

        self.data_module = data_module

        # Non-tokenized dataframe
        self.raw_dataframe = self.get_statistics()

        # tokenize and create the vocabulary
        self.vocabulary_dataframe, tokenized_dataframe_no_max_label_length = self.run_tokenizer()

        # pass the max_label_length
        self.pretokenized_dataframe = tokenized_dataframe_no_max_label_length[tokenized_dataframe_no_max_label_length['tokenized_len'] < self.data_module.set_max_label_length]
        self.tokenized_dataframe = self.pretokenized_dataframe[0:data_module.number_png_images_to_use_in_dataset]

        # pass the max width:
        self.tokenized_dataframe = self.tokenized_dataframe[self.tokenized_dataframe['width'] <= self.data_module.max_width]
        #self.tokenized_dataframe = self.tokenized_dataframe[(self.tokenized_dataframe['width'] >0 ) & (self.tokenized_dataframe['height'] >0 )]

        self.max_label_length =  data_module.set_max_label_length + 2 # accounting for the Start and End Tokens
        self.vocabulary = create_vocabulary_dictionary_from_dataframe(self.vocabulary_dataframe)

        self.inverse_vocabulary = invert_vocabulary(self.vocabulary)


    ########## Methods to generate Pandas DataFrame, create a vocabulary and Tokenize #########

    def get_statistics(self):
        # get dataframe
        formulas_df = _get_dataframe()

        return _get_stats(formulas_df)


    # creates a vocabulary of tokes used for further processing
    def run_tokenizer(self):
        return make_vocabulary(self.get_statistics())





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


