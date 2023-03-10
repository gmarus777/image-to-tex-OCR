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

WIDTH = 2048
HEIGHT = 1024


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
        self.pretokenized_dataframe = tokenized_dataframe_no_max_label_length[tokenized_dataframe_no_max_label_length['tokenized_len'] < data_module.set_max_label_length]
        self.tokenized_dataframe_pre_resize = self.pretokenized_dataframe[0:data_module.number_png_images_to_use_in_dataset]

        # to cut longer formulas remove next line and add the following
        # self.tokenized_dataframe = self.tokenized_dataframe_pre_resize
        self.tokenized_dataframe = self.tokenized_dataframe_pre_resize[(self.tokenized_dataframe_pre_resize['width']<WIDTH) & (self.tokenized_dataframe_pre_resize['height']<HEIGHT)]

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

    # PRINTED
    path_to_formulas = PrintedLatexDataConfig.PNG_FINAL_FORMULAS
    formulas_df = readlines_to_df(path_to_formulas, 'formula')


    # get printed png image names
    image_names_path = PrintedLatexDataConfig.PNG_IMAGES_NAMES_FILE
    image_names_df = readlines_to_df(image_names_path, 'image_name')

    formulas_df['image_name'] = image_names_df


    # HANDWRITTEN

    path_to_hw_formulas = PrintedLatexDataConfig.HANDWRITTEN_TRAIN


    path_to_formulas_hw = PrintedLatexDataConfig.HANDWRITTEN_FORMULAS

    images_df_hw, formula_locations_hw = readlines_to_df_images_and_list( path_to_list= path_to_hw_formulas)
    formulas_df_hw = readlines_to_df_formulas(formula_locations = formula_locations_hw, path = path_to_formulas_hw)
    formulas_df_hw['image_name'] = images_df_hw

    # final_formulas = pd.concat([formulas_df,formulas_df_hw], ignore_index=True)

    final_formulas = formulas_df





    return final_formulas


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


def readlines_to_df_images_and_list(path_to_list):
    formula_locations = []
    rows_images = []

    n = 0
    with open(path_to_list, 'r') as file_train_list:
        for line in file_train_list.readlines():
            line.strip()

            l = line.split(' ')

            formula_line = int(l[0])
            image_name = l[1] + '.png'
            rows_images.append(image_name)
            formula_locations.append(formula_line)

    images_df = pd.DataFrame({'image_name': rows_images}, dtype=np.str_)

    return images_df, formula_locations

def readlines_to_df_formulas(formula_locations, path, ):
    rows_formulas = []

    # obtain the corresponding formula

    formulas = open(path).read().split('\n')


    for formula_id in formula_locations:

        formula = formulas[formula_id]



        rows_formulas.append(formula)

    formulas_df = pd.DataFrame({'formula': rows_formulas},dtype=str) #dtype=np.str_


    return formulas_df