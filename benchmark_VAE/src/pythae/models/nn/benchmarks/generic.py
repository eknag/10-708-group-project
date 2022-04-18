"""Proposed Neural nets architectures suited for MNIST"""

import torch
import torch.nn as nn

from typing import List
from ..base_architectures import BaseEncoder, BaseDecoder
from ...base.base_utils import ModelOutput
from ... import BaseAEConfig

from pythae.models.nn import BaseEncoder, BaseDecoder, BaseDiscriminator

from functools import lru_cache
import numpy as np


def get_encoder_layers():
    layers = nn.ModuleList()

    layers = nn.ModuleList()

    layers.append(
        nn.Sequential(
            nn.Conv2d(self.n_channels, 128, 4, 2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
    )

    layers.append(
        nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
    )

    layers.append(
        nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
    )

    layers.append(
        nn.Sequential(
            nn.Conv2d(512, 1024, 4, 2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
    )

    return layers


@lru_cache(maxsize=None)
def get_conv_output_dimension(input_dim: tuple, layers: List[nn.Module]) -> tuple:
    """
    Get the output dimension of the convolutional layers.
    """
    input_sample = torch.zeros(1, *input_dim)
    for layer in layers:
        input_sample = layer(input_sample)

    return input_sample.shape[1:]


class Encoder_AE_GENERIC(BaseEncoder):
    """
    A Generic Convolutional encoder Neural net suited for Autoencoder-based models.

    It can be built as follows:

    .. code-block::

            >>> from pythae.models.nn.benchmarks.celeba import Encoder_AE_GENERIC
            >>> from pythae.models import AEConfig
            >>> model_config = AEConfig(input_dim=(3, 64, 64), latent_dim=64)
            >>> encoder = Encoder_AE_GENERIC(model_config)


    .. note::

        Please note that this encoder is only suitable for Autoencoder based models since it only
        outputs the embeddings of the input data under the key `embedding`.

    """

    def __init__(self, args: BaseAEConfig):
        BaseEncoder.__init__(self)

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.n_channels = 3

        self.layers = get_encoder_layers()
        self.depth = len(self.layers)

        self.conv_out_size = get_conv_output_dimension(self.input_dim, self.layers)

        self.embedding = nn.Linear(np.prod(self.conv_out_size), args.latent_dim)

    def forward(self, x: torch.Tensor, output_layer_levels: List[int] = None):
        """Forward method

        Args:
            output_layer_levels (List[int]): The levels of the layers where the outputs are
                extracted. If None, the last layer's output is returned. Default: None.

        Returns:
            ModelOutput: An instance of ModelOutput containing the embeddings of the input data
            under the key `embedding`. Optional: The outputs of the layers specified in
            `output_layer_levels` arguments are available under the keys `embedding_layer_i` where
            i is the layer's level."""
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:

            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth}). "
                f"Got ({output_layer_levels})."
            )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = x

        for i in range(max_depth):
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"embedding_layer_{i+1}"] = out
            if i + 1 == self.depth:
                output["embedding"] = self.embedding(out.reshape(x.shape[0], -1))

        return output


class Encoder_VAE_GENERIC(BaseEncoder):
    """
    A Convolutional encoder Neural net suited for GENERIC-64 and
    Variational Autoencoder-based models.

    It can be built as follows:

    .. code-block::

            >>> from pythae.models.nn.benchmarks.generic import Encoder_VAE_GENERIC
            >>> from pythae.models import VAEConfig
            >>> model_config = VAEConfig(input_dim=(3, 64, 64), latent_dim=64)
            >>> encoder = Encoder_VAE_GENERIC(model_config)


    and then passed to a :class:`pythae.models` instance

        >>> from pythae.models import VAE
        >>> model = VAE(model_config=model_config, encoder=encoder)
        >>> model.encoder == encoder
        ... True


    .. note::

        Please note that this encoder is only suitable for Variational Autoencoder based models
        since it outputs the embeddings and the **log** of the covariance diagonal coefficients
        of the input data under the key `embedding` and `log_covariance`.


    """

    def __init__(self, args: BaseAEConfig):
        BaseEncoder.__init__(self)

        raise NotImplementedError
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.n_channels = 3

        self.layers = get_encoder_layers()
        self.depth = len(self.layers)

        self.conv_out_size = get_conv_output_dimension(self.input_dim, self.layers)

        self.embedding = nn.Linear(np.prod(self.conv_out_size), args.latent_dim)
        self.log_var = nn.Linear(np.prod(self.conv_out_size), args.latent_dim)

    def forward(self, x: torch.Tensor, output_layer_levels: List[int] = None):
        """Forward method

        Args:
            output_layer_levels (List[int]): The levels of the layers where the outputs are
                extracted. If None, the last layer's output is returned. Default: None.

        Returns:
            ModelOutput: An instance of ModelOutput containing the embeddings of the input data
            under the key `embedding` and the **log** of the diagonal coefficient of the covariance
            matrices under the key `log_covariance`. Optional: The outputs of the layers specified
            in `output_layer_levels` arguments are available under the keys `embedding_layer_i`
            where i is the layer's level.
        """
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:

            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth}). "
                f"Got ({output_layer_levels})."
            )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = x

        for i in range(max_depth):
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"embedding_layer_{i+1}"] = out

            if i + 1 == self.depth:
                output["embedding"] = self.embedding(out.reshape(x.shape[0], -1))
                output["log_covariance"] = self.log_var(out.reshape(x.shape[0], -1))

        return output


class Decoder_AE_GENERIC(BaseDecoder):
    """
    A Convolutional decoder Neural net suited for GENERIC-64 and Autoencoder-based
    models.

    It can be built as follows:

    .. code-block::

            >>> from pythae.models.nn.benchmarks.generic import Decoder_AE_GENERIC
            >>> from pythae.models import VAEConfig
            >>> model_config = VAEConfig(input_dim=(3, 64, 64), latent_dim=64)
            >>> decoder = Decoder_AE_GENERIC(model_config)


    and then passed to a :class:`pythae.models` instance

        >>> from pythae.models import VAE
        >>> model = VAE(model_config=model_config, decoder=decoder)
        >>> model.decoder == decoder
        ... True

    .. note::

        Please note that this decoder is suitable for **all** models.

    """

    def __init__(self, args: dict):
        BaseDecoder.__init__(self)
        self.input_dim = (3, 64, 64)
        self.latent_dim = args.latent_dim
        self.n_channels = 3

        layers = nn.ModuleList()

        layers.append(nn.Sequential(nn.Linear(args.latent_dim, 1024 * 8 * 8)))

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(1024, 512, 5, 2, padding=2),
                nn.BatchNorm2d(512),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, 5, 2, padding=1, output_padding=0),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, 5, 2, padding=2, output_padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(128, self.n_channels, 5, 1, padding=1),
                nn.Sigmoid(),
            )
        )

        self.layers = layers
        self.depth = len(self.layers)

    def forward(self, z: torch.Tensor, output_layer_levels: List[int] = None):
        """Forward method

        Args:
            output_layer_levels (List[int]): The levels of the layers where the outputs are
                extracted. If None, the last layer's output is returned. Default: None.

        Returns:
            ModelOutput: An instance of ModelOutput containing the reconstruction of the latent code
            under the key `reconstruction`. Optional: The outputs of the layers specified in
            `output_layer_levels` arguments are available under the keys `reconstruction_layer_i`
            where i is the layer's level.
        """
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:

            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth}). "
                f"Got ({output_layer_levels})."
            )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = z

        for i in range(max_depth):
            out = self.layers[i](out)

            if i == 0:
                out = out.reshape(z.shape[0], 1024, 8, 8)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"reconstruction_layer_{i+1}"] = out

            if i + 1 == self.depth:
                output["reconstruction"] = out

        return output
