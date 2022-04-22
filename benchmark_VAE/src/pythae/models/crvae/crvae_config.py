from pydantic.dataclasses import dataclass

from ..base.base_config import BaseAEConfig


@dataclass
class CRVAEConfig(BaseAEConfig):
    """This is the autoencoder model configuration instance deriving from
    :class:`~BaseAEConfig`.

    Parameters:
        input_dim (int): The input_data dimension
        latent_dim (int): The latent space dimension. Default: None.
        aug_type (str): The name of the transform function. Default: None. See AugmentationProcessor for list of names
        gamma (float): The gamma parameter for cr_loss. Default: 0.001
        beta_1 (float): The beta_1 parameter for elbo_loss_1. Default: 1
        beta_2 (float): The beta_2 parameter for elbo_loss_2. Default: 1
        default_encoder (bool): Whether the encoder default. Default: True.
        default_decoder (bool): Whether the encoder default. Default: True.
    """

    aug_type: str = None
    gamma: float = 0.001
    beta_1: float = 1
    beta_2: float = 1
    reconstruction_loss = "mse"
