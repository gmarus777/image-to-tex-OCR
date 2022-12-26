import torch
import os
from pathlib import Path
from transformers import PreTrainedTokenizerFast
from tokenizers.processors import TemplateProcessing
from tokenizers import Tokenizer, pre_tokenizers, models, decoders, trainers, processors

from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


SPECIAL_TOKENS= ["<S>", "<E>", "<P>"]


PNG_FORMULA_FILE = 'Data/processed_data/final_png_formulas.txt'
NON_TOKENIZED_FORMULAS = 'Data/processed_data/normalized/formulas.txt'

TOKENIZER_JSON = 'Data/processed_data/tokenizer.json'

class Label_Transforms:


    def __init__(self,
                labels_transform_name = None,
                vocabulary = None,
                max_label_length = None,
                BPE_set_vocab_size = None):



        self.vocabulary = vocabulary
        self.max_label_length = max_label_length


        if labels_transform_name.lower() == 'bpe':

            # Generate the tokenizer json by parsing PNG_FORMULA_FILE, outputs TOKENIZER_JSON
            generate_tokenizer(equations=[NON_TOKENIZED_FORMULAS],
                               output = TOKENIZER_JSON,
                               vocab_size = BPE_set_vocab_size)

            # Initiating the Tokenizer
            # First we create a tokenizer object from json
            tokenizer_obj = Tokenizer.from_file(TOKENIZER_JSON)

            # we add prost-processing ( add start and end tokens)
            tokenizer_obj.post_processor = TemplateProcessing(single="<S> $A <E>",
                                                              special_tokens=[("<S>", 0), ("<E>", 1)])
            tokenizer_obj.enable_padding(pad_id=2, pad_token='<P>', length= self.max_label_length)
            # tokenizer_obj.enable_truncation(max_length = self.max_label_length)

            # we pass the tokenizer object into PreTrainedTokenizerFast and initiate it
            self.tokenizer = PreTrainedTokenizerFast(model_max_length = self.max_label_length,
                                                bos_token='<S>',
                                                eos_token='<E>',
                                                pad_token='<P>',
                                                add_special_tokens=True,
                                                tokenizer_object=tokenizer_obj
                                                )

            self.vocab_size = self.tokenizer.vocab_size
            self.vocabulary = self.tokenizer.vocab

    def bpe_convert_strings_to_labels(self, string) -> torch.Tensor:

        # we create a string for conversion from list

        new_string = ''.join(string)


        output = self.tokenizer.encode(text = new_string,
                                  pad_to_max_length=True,
                                  max_length =self.max_label_length,
                                       truncation=True,
                                  return_tensors='pt')
        return output






    def convert_strings_to_labels(self, string ) -> torch.Tensor:
        """
        Converts a string to a ( length) ndarray, with each string wrapped with <S> and <E> tokens,
        and padded with the <P> token.
        """

        # first create a (1, length) tensor of padding
        labels = torch.ones((self.max_label_length), dtype=torch.long) * self.vocabulary["<P>"]
        tokens = list(string)
        tokens = ["<S>", *tokens, "<E>"]

        for i, token in enumerate(tokens):
            labels[i] = self.vocabulary[token]

        return labels


def generate_tokenizer(equations, output, vocab_size):


    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(special_tokens=SPECIAL_TOKENS, vocab_size = vocab_size, show_progress=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train(equations, trainer)
    tokenizer.save(path = output, pretty=False)





