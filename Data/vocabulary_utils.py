import pandas as pd
import json
import torch


START_TOKEN ="<S>"
END_TOKEN = "<E>"
PADDING_TOKEN = "<P>"






class TokenDict(object):
    def __init__(self):
        self._tokens = {}

    def account(self, token_list):
        for token in token_list:
            self._count(token)

    def _count(self, token):
        if token in self._tokens:
            self._tokens[token] += 1
        else:
            self._tokens[token] = 1
        return 1

    @property
    def dict(self):
        return self._tokens

    @property
    def tokens(self):
        return sorted(self._tokens.keys())


def append_special_words(df_vocab_, freq_):
    assert 0 not in df_vocab_.id.values
    df_vocab_ = df_vocab_.append(pd.DataFrame({'id': 0, 'freq': freq_}, index=[r'\eos']), verify_integrity=True)
    assert 1 not in df_vocab_.id.values
    df_vocab_ = df_vocab_.append(pd.DataFrame({'id': 1, 'freq': freq_}, index=[r'\bos']), verify_integrity=True)
    return df_vocab_


def remove_special_words(df_vocab_):
    return df_vocab_.drop(labels=[r'\eos', r'\bos'])


def make_vocabulary(df_):

    ## Assume that the latex formula strings are already tokenized into string-tokens separated by whitespace
    ## Hence we just need to split the string by whitespace.
    sr_token = df_.formula.apply(str).str.split(' ')

    sr_tokenized_len = sr_token.str.len()
    df_tokenized = df_.assign(latex_tokenized=sr_token, tokenized_len=sr_tokenized_len)
    ## Aggregate the tokens
    vocab = TokenDict()
    sr_token.agg(lambda l: vocab.account(l))
    ## Sort and save
    tokens = []
    count = []

    # we add special tokens 0:  and  1:
    tokens.append(START_TOKEN)
    tokens.append(END_TOKEN)
    tokens.append(PADDING_TOKEN)
    count.append(0)
    count.append(0)
    count.append(0)
    for t in vocab.tokens:
        tokens.append(t)
        count.append(vocab.dict[t])

    ## Assign token-ids. Start with 3 (space token) and reserve 0 and 1 for EOS and BOS added later in label transform.
    df_vocab = pd.DataFrame({'token': tokens, 'id': range( len(tokens) ), 'freq': count}, columns=['token','id', 'freq'])
    # df_vocab = append_special_words(df_vocab, df_.shape[0])
    #print('Vocab Size = ', df_vocab.shape[0])
    max_id = df_vocab.id.max()
    #print('Max TokenID = ', max_id, type(max_id))



    #display(df_tokenized.iloc[:1])
    return df_vocab, df_tokenized



def create_vocabulary_dictionary_from_dataframe(dataframe):
    vocabulary = {}

    for i in range(len(dataframe)):
        vocabulary[dataframe.loc[i][0]] = dataframe.loc[i][1]


    return vocabulary







def convert_strings_to_labels(string, vocabulary, length ) -> torch.Tensor:
    """
    Converts a string to a ( length) ndarray, with each string wrapped with <S> and <E> tokens,
    and padded with the <P> token.
    """


    # first create a (1, length) tensor of padding
    labels = torch.ones((length), dtype=torch.long) * vocabulary["<P>"]
    tokens = list(string)
    tokens = ["<S>", *tokens, "<E>"]


    for i, token in enumerate(tokens):
        labels[i] = vocabulary[token]

    return labels


def invert_vocabulary(vocabulary):
    inverse_vocabulary = {}
    for letter, idx in vocabulary.items():
        inverse_vocabulary[idx] = letter
    return inverse_vocabulary

def load_dic(filename):
    with open(filename) as f:
        dic = json.loads(f.read())
    return dic

# NOT USED
def generate_character_tokenizer(vocabulary_path):
    with open(CHARACTER_VOCABULARY_PATH) as f:
        essentials = json.load(f)
        mapping = list(essentials["characters"])
        inverse_mapping = {v: k for k, v in enumerate(mapping)}

    return mapping, inverse_mapping