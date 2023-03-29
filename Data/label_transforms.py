import torch



SPECIAL_TOKENS= ["<S>", "<E>", "<P>"]



class Label_Transforms:


    def __init__(self,
                labels_transform_name = None,
                vocabulary = None,
                max_label_length = None,
                 ):



        self.vocabulary = vocabulary
        self.max_label_length = max_label_length








    def convert_strings_to_labels(self, string, length=None ) -> torch.Tensor:
        """
        Converts a string to a ( length) ndarray, with each string wrapped with <S> and <E> tokens,
        and padded with the <P> token.
        """

        # first create a (1, length) tensor of padding
        if length is None:
            labels = torch.ones((self.max_label_length), dtype=torch.long) * self.vocabulary["<P>"]
        else:
            labels = torch.ones((length), dtype=torch.long) * self.vocabulary["<P>"]
        tokens = list(string)
        tokens = ["<S>", *tokens, "<E>"]

        for i, token in enumerate(tokens):
            labels[i] = self.vocabulary[token]

        return labels








