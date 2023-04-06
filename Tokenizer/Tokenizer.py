import json
import os

class Tokenizer:
    def __init__(self):
        self.vocabulary = self.invert_vocabulary(self.load_dic('Tokenizer/230k_ver2.json'))



    def load_dic(self, filename):
        with open(filename) as f:
            dic = json.loads(f.read())
            dic_new = dict((k, int(v)) for k, v in dic.items())
        return dic_new


    def invert_vocabulary(self, vocabulary):
        inverse_vocabulary = {}
        for letter, idx in vocabulary.items():
            inverse_vocabulary[idx] = letter
        return inverse_vocabulary



def token_to_strings( tokens):
    skipTokens = {'<S>', '<E>', '<P>'}
    inverse_mapping = Tokenizer().vocabulary
    s = ''
    if tokens.shape[0] == 1:
        tokens = tokens[0]
    for number in tokens:
        letter = inverse_mapping[number.item()]
        if letter not in skipTokens:
            s = s + " " + str(letter)
    return s