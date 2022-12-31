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
            # token.encode("utf-8")
            # token = token.decode("utf-8")
            labels[i] = self.vocabulary[token]

        return labels


    def convert_strings_to_labels_HW(self, string ) -> torch.Tensor:
        """
        Converts a string to a ( length) ndarray, with each string wrapped with <S> and <E> tokens,
        and padded with the <P> token.
        """

        # first create a (1, length) tensor of padding
        labels = torch.ones((self.max_label_length), dtype=torch.long) * self.vocabulary["<P>"]
        tokens = list(string)
        tokens = ["<S>", *tokens, "<E>"]

        for i, token in enumerate(tokens):
            # token.encode("utf-8")
            # token = token.decode("utf-8")
            try:
                labels[i] = self.vocabulary[token]
            except:

                if token == "\Delta":
                    print('one delta')
                    labels[i] = self.vocabulary["\\delta"]

                if token == '\\operatorname':
                    labels[i] = self.vocabulary["\mathrm"]

                if token == '\operatorname':
                    labels[i] = self.vocabulary["\mathrm"]

                if token == '\\operatorname*':
                    labels[i] = self.vocabulary["\mathrm"]

                if token == '\operatorname*':
                    labels[i] = self.vocabulary["\mathrm"]

                if token == "\\Delta":
                    print('two delta')
                    labels[i] = self.vocabulary["\\delta"]

                if token == '\hspace':
                    labels[i] = self.vocabulary["\mathrm"]

                if token == '\\hspace':
                    labels[i] = self.vocabulary["\mathrm"]

                else:
                    print(token)
                    labels[i] = self.vocabulary["?"]

        return labels








