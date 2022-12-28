import math
from typing import Union

import torch
import torch.nn as nn
import torchvision.models
from torch import Tensor
from Models.positional_encoding import PositionalEncoding1D, PositionalEncoding2D



TF_DIM = 128    # embedding_dim
TF_FC_DIM = 256 # decoder fully connected dim
TF_DROPOUT = 0.3 # decoder_dropout
TF_LAYERS = 4   # decoder_layers
TF_NHEAD = 8    # decoder_heads
RESNET_DIM = 512  # hard-coded



# TODO: Pass parameters from the dataset

# self.d_model = TF_DIM
# self.max_output_len = dataset.max_label_length
# dim_feedforward = TF_FC_DIM


'''
Image to Tex OCR:
Resnet18 encoder with the first three layers and Transformer Decoder.

'''
class ResNetTransformer(nn.Module):
    def __init__(
        self,
        dataset = None,
        max_label_length = None,
        num_classes = None,
        embedding_dim = TF_DIM,
        decoder_heads = TF_NHEAD,
        decoder_layers = TF_LAYERS,
        decoder_dropout = TF_DROPOUT,
        decoder_fc = TF_FC_DIM,
    ) -> None:

        super().__init__()

        self.embedding_dim = embedding_dim

        # Set maximum label length manually of from a dataset
        # if max_label_length is not None:
            # self.max_output_len = max_label_length
        # else:

        self.max_output_len = dataset.max_label_length

        self.sos_index = int(0) # int(dataset.vocabulary['<S>'])
        self.eos_index =  int(1) # int(dataset.vocabulary['<E>'])
        self.pad_index =  int(2) # int(dataset.vocabulary['<P>'])
        self.num_classes =int(len(dataset.vocabulary))


        ### Encoder ###
        resnet = torchvision.models.resnet18(pretrained=False)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )
        self.bottleneck = nn.Conv2d(256, self.embedding_dim, 1) # in channels, out channels, stride
        self.image_positional_encoder = PositionalEncoding2D(self.embedding_dim)

        ### Decoder ###
        self.embedding = nn.Embedding(self.num_classes, self.embedding_dim)
        self.y_mask = generate_square_subsequent_mask(self.max_output_len)
        self.word_positional_encoder = PositionalEncoding1D(self.embedding_dim, max_len=self.max_output_len)
        transformer_decoder_layer = nn.TransformerDecoderLayer(self.embedding_dim, decoder_heads, decoder_fc, decoder_dropout)
        self.transformer_decoder = nn.TransformerDecoder(transformer_decoder_layer, decoder_layers)
        self.fc = nn.Linear(self.embedding_dim, self.num_classes)


        # It is empirically important to initialize weights properly
        if self.training:
            self._init_weights()


    def _init_weights(self) -> None:
        """Initialize weights."""
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-init_range, init_range)

        nn.init.kaiming_normal_(
            self.bottleneck.weight.data,
            a=0,
            mode="fan_out",
            nonlinearity="relu",
        )
        if self.bottleneck.bias is not None:
            _, fan_out = nn.init._calculate_fan_in_and_fan_out(self.bottleneck.weight.data)
            bound = 1 / math.sqrt(fan_out)
            nn.init.normal_(self.bottleneck.bias, -bound, bound)


    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: (B, _E, _H, _W)
            y: (B, Sy) with elements in (0, num_classes - 1)

        Returns:
            (B, num_classes, Sy) logits
        """
        encoded_x = self.encode(x)  # (Sx, B, E)
        output = self.decode(y, encoded_x)  # (Sy, B, num_classes)
        output = output.permute(1, 2, 0)  # (B, num_classes, Sy)
        return output

    def encode(self, x: Tensor) -> Tensor:
        """Encode inputs.

        Args:
            x: (B, C, _H, _W)

        Returns:
            (Sx, B, E)
        """
        # Resnet expects 3 channels but training images are in gray scale
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.backbone(x.float())  # (B, RESNET_DIM, H, W); H = _H // 32, W = _W // 32
        x = self.bottleneck(x)  # (B, E, H, W)
        x = self.image_positional_encoder(x)  # (B, E, H, W)
        x = x.flatten(start_dim=2)  # (B, E, H * W)
        x = x.permute(2, 0, 1)  # (Sx, B, E); Sx = H * W
        return x

    def decode(self, y: Tensor, encoded_x: Tensor) -> Tensor:
        """Decode encoded inputs with teacher-forcing.

        Args:
            encoded_x: (Sx, B, E)
            y: (B, Sy) with elements in (0, num_classes - 1)

        Returns:
            (Sy, B, num_classes) logits
        """
        y = y.permute(1, 0)  # (Sy, B)
        y = self.embedding(y) * math.sqrt(self.embedding_dim)  # (Sy, B, E)
        y = self.word_positional_encoder(y)  # (Sy, B, E)
        Sy = y.shape[0]
        y_mask = self.y_mask[:Sy, :Sy].type_as(encoded_x)  # (Sy, Sy)
        output = self.transformer_decoder(y, encoded_x, y_mask)  # (Sy, B, E)
        output = self.fc(output)  # (Sy, B, num_classes)
        return output

    def predict(self, x: Tensor) -> Tensor:
        """Make predctions at inference time.

        Args:
            x: (B, C, H, W). Input images.

        Returns:
            (B, max_output_len) with elements in (0, num_classes - 1).
        """
        B = x.shape[0]
        S = self.max_output_len

        encoded_x = self.encode(x)  # (Sx, B, E)

        output_indices = torch.full(size=(B, S), fill_value=self.pad_index).type_as(x).long()
        output_indices[:, 0] = self.sos_index
        has_ended = torch.full((B,), False)

        for Sy in range(1, S):
            y = output_indices[:, :Sy]  # (B, Sy)
            logits = self.decode(y, encoded_x)  # (Sy, B, num_classes)
            # Select the token with the highest conditional probability
            output = torch.argmax(logits, dim=-1)  # (Sy, B)
            output_indices[:, Sy] = output[-1:]  # Set the last output token

            # Early stopping of prediction loop to speed up prediction
            has_ended |= (output_indices[:, Sy] == self.eos_index).type_as(has_ended)
            if torch.all(has_ended):
                break

        # Set all tokens after end token to be padding
        eos_positions = find_first(output_indices, self.eos_index)
        for i in range(B):
            j = int(eos_positions[i].item()) + 1
            output_indices[i, j:] = self.pad_index

        return output_indices


def generate_square_subsequent_mask(size: int) -> Tensor:
    """Generate a triangular (size, size) mask."""
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    return mask


def find_first(x: Tensor, element: Union[int, float], dim: int = 1) -> Tensor:
    """Find the first occurence of element in x along a given dimension.

    Args:
        x: The input tensor to be searched.
        element: The number to look for.
        dim: The dimension to reduce.

    Returns:
        Indices of the first occurence of the element in x. If not found, return the
        length of x along dim.

    Usage:
        >>> first_element(Tensor([[1, 2, 3], [2, 3, 3], [1, 1, 1]]), 3)
        tensor([2, 1, 3])

    Reference:
        https://discuss.pytorch.org/t/first-nonzero-index/24769/9

        I fixed an edge case where the element we are looking for is at index 0. The
        original algorithm will return the length of x instead of 0.
    """
    mask = x == element
    found, indices = ((mask.cumsum(dim) == 1) & mask).max(dim)
    indices[(~found) & (indices == 0)] = x.shape[dim]
    return indices




