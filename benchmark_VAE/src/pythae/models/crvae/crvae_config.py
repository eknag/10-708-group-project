from pydantic.dataclasses import dataclass

from ..base.base_config import BaseAEConfig


@dataclass
class CRVAEConfig(BaseAEConfig):
    """This is the autoencoder model configuration instance deriving from
    :class:`~BaseAEConfig`.

    Parameters:
        input_dim (int): The input_data dimension
        latent_dim (int): The latent space dimension. Default: None.
        transform (str): The name of the transform function. Default: None. See AugmentationProcessor for list of names
        default_encoder (bool): Whether the encoder default. Default: True.
        default_decoder (bool): Whether the encoder default. Default: True.
    """

    reconstruction_loss = "mse"
