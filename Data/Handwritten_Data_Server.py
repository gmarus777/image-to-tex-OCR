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
from Data.vocabulary_utils import create_vocabulary_dictionary_from_dataframe, invert_vocabulary
import pandas as pd
import json



START_TOKEN ="<S>"
END_TOKEN = "<E>"
PADDING_TOKEN = "<P>"


class Handwritten_Data_Server:
    def __init__(self,
                 data_module = None,
                 ):

        self.data_module = data_module

        # Non-tokenized dataframe
        self.raw_dataframe = self.get_statistics()

        # tokenize and create the vocabulary
        tokenized_dataframe_no_max_label_length = self.run_tokenizer()

        # pass the max_label_length
        self.pretokenized_dataframe = tokenized_dataframe_no_max_label_length[tokenized_dataframe_no_max_label_length['tokenized_len'] < data_module.set_max_label_length]
        self.tokenized_dataframe = self.pretokenized_dataframe[0:data_module.number_png_images_to_use_in_dataset]

        self.max_label_length =  data_module.set_max_label_length + 2 # accounting for the Start and End Tokens
        self.vocabulary = load_dic('Data/Data_Bank/258_Test_run.json')

        self.inverse_vocabulary = invert_vocabulary(self.vocabulary)


    ########## Methods to generate Pandas DataFrame, create a vocabulary and Tokenize #########

    def get_statistics(self):
        # get dataframe
        formulas_df = _get_dataframe()

        return _get_stats(formulas_df)


    # creates a vocabulary of tokes used for further processing
    def run_tokenizer(self):
        return make_vocabulary(self.raw_dataframe)





####### Helper Functions for Pandas DataFrame generation ###########

def _get_dataframe():

    # take final formula list
    path_to_train = PrintedLatexDataConfig.HANDWRITTEN_TRAIN
    path_to_val = PrintedLatexDataConfig.HANDWRITTEN_VAL

    # get png image names nad formula line location
    path_to_formulas = PrintedLatexDataConfig.HANDWRITTEN_FORMULAS

    # formulas_df = readlines_to_df(path_to_list=path_to_val ,path =path_to_formulas, colname='formula', colname_im = 'image_name')
    images_df, formula_locations = readlines_to_df_images_and_list( path_to_list= path_to_train)
    formulas_df = readlines_to_df_formulas(formula_locations = formula_locations, path = path_to_formulas)
    formulas_df['image_name'] = images_df


    return formulas_df


# outputs formula length, image height and width.
def _get_stats(datasetDF):
    widths = []
    heights = []
    formula_lens = []

    dataset = datasetDF
    for _, row in datasetDF.iterrows():
        image_name = row.image_name

        im = Image.open(os.path.join(PrintedLatexDataConfig.HANDWRITTEN_IMAGES_FOLDER, image_name))
        widths.append(im.size[0])
        heights.append(im.size[1])
        formula_lens.append(len(row.formula))

    # datasetDF = datasetDF.assign(width=widths, height=heights, formula_len=formula_lens)
    dataset['height'] = heights
    dataset['width'] = widths
    dataset['formula_length'] = formula_lens


    return dataset








# converts formulas txt to pandas dataframe
def readlines_to_df(path_to_list, path, colname, colname_im):
    rows_formulas = []
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

    # obtain the corresponding formula
    with open(path) as file_formulas:
        for formula_id in formula_locations:
            file_formulas.seek(0)
            formula = file_formulas.readlines()[formula_id]
            formula.strip()
            rows_formulas.append(formula)

    formulas_df = pd.DataFrame({colname: rows_formulas}, dtype=np.str_)
    images_df = pd.DataFrame({colname_im: rows_images}, dtype=np.str_)
    formulas_df['image_name'] = images_df


    return formulas_df


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

    formulas_df = pd.DataFrame({'formula': rows_formulas}, dtype=np.str_)


    return formulas_df


def load_dic(filename):
    with open(filename) as f:
        dic = json.loads(f.read())
        dic_new = dict((k, int(v)) for k, v in dic.items())
    return dic_new


def make_vocabulary(df_):

    ## Assume that the latex formula strings are already tokenized into string-tokens separated by whitespace
    ## Hence we just need to split the string by whitespace.
    sr_token = df_.formula.apply(str).str.split(' ')

    sr_tokenized_len = sr_token.str.len()
    df_tokenized = df_.assign(latex_tokenized=sr_token, tokenized_len=sr_tokenized_len)



    return  df_tokenized
