import torch
import os

from pythae.models.base import BaseAE
from pythae.models.crvae.crvae_config import CRVAEConfig
from pythae.data.datasets import BaseDataset
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseEncoder, BaseDecoder
from pythae.models.nn.default_architectures import Encoder_VAE_MLP

from typing import Optional
from torchvision import transforms

import torch.nn.functional as F
import random


class CRVAE(BaseAE):
    """Vanilla Variational Autoencoder model.

    Args:
        model_config(CRVAEConfig): The Variational Autoencoder configuration seting the main
        parameters of the model
        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.
        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.
    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(
        self,
        model_config: CRVAEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        BaseAE.__init__(self, model_config=model_config, decoder=decoder)

        self.model_name = "CRVAE"

        if encoder is None:
            if model_config.input_dim is None:
                raise AttributeError(
                    "No input dimension provided !"
                    "'input_dim' parameter of BaseAEConfig instance must be set to 'data_shape' where "
                    "the shape of the data is (C, H, W ..). Unable to build encoder "
                    "automatically"
                )

            encoder = Encoder_VAE_MLP(model_config)
            self.model_config.uses_default_encoder = True

        else:
            self.model_config.uses_default_encoder = False

        self.set_encoder(encoder)

        self.aug_type = model_config.aug_type
        self.gamma = model_config.gamma
        self.beta_1 = model_config.beta_1
        self.beta_2 = model_config.beta_2

    def forward(self, inputs: BaseDataset, **kwargs):
        """
        The VAE model
        Args:
            inputs (BaseDataset): The training datasat with labels
        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """

        x = inputs["data"]
        n = x.size(0)
        tx = self.apply_transform(x, self.aug_type)
        #   x = torch.cat([x, tx], dim=0)

        encoder_output_x = self.encoder(x)
        encoder_output_tx = self.encoder(tx)

        mu_x, log_var_x = encoder_output_x.embedding, encoder_output_x.log_covariance
        mu_tx, log_var_tx = (
            encoder_output_tx.embedding,
            encoder_output_tx.log_covariance,
        )

        std_x = torch.exp(0.5 * log_var_x)
        z_x, eps_x = self._sample_gauss(mu_x, std_x)
        recon_x = self.decoder(z_x)["reconstruction"]

        #   mu_x, mu_tx = mu_x[:n], mu_x[n:]
        #   log_var_x, log_var_tx = log_var_x[:n], log_var_x[n:]
        #   z_x, z_tx = z_x[:n], z_x[n:]
        #   x, tx = x[:n], x[n:]
        #   recon_x, recon_tx = recon_x[:n], recon_x[n:]

        std_tx = torch.exp(0.5 * log_var_tx)
        z_tx, eps_tx = self._sample_gauss(mu_tx, std_tx)
        recon_tx = self.decoder(z_tx)["reconstruction"]

        loss, recon_loss, kld = self.loss_function(
            recon_x, recon_tx, x, tx, mu_x, mu_tx, log_var_x, log_var_tx, z_x, z_tx
        )

        output = ModelOutput(
            reconstruction_loss=recon_loss,
            reg_loss=kld,
            loss=loss,
            recon_x=recon_x,
            z=z_x,
        )

        return output

    def loss_function(
        self, recon_x, recon_tx, x, tx, mu_x, mu_tx, log_var_x, log_var_tx, z_x, z_tx
    ):
        vae_loss_x, bce_x, kld_x = self._elbo_loss(
            mu_x, log_var_x, recon_x, x, self.beta_1
        )
        vae_loss_tx, bce_tx, kld_tx = self._elbo_loss(
            mu_tx, log_var_tx, recon_tx, tx, self.beta_2
        )
        cr_vae_loss = self._cr_loss(mu_x, log_var_x, mu_tx, log_var_tx, self.gamma)
        return vae_loss_x + vae_loss_tx + cr_vae_loss, bce_x + bce_tx, kld_x + kld_tx

    def _elbo_loss(self, mu, logvar, recon, original, beta):
        batch_size = original.size(0)
        if self.model_config.reconstruction_loss == "mse":
            recon_loss = F.mse_loss(
                recon.reshape(batch_size, -1),
                original.reshape(batch_size, -1),
                reduction="none",
            ).sum(dim=-1)
        elif self.model_config.reconstruction_loss == "bce":
            recon_loss = F.binary_cross_entropy(
                recon.reshape(batch_size, -1),
                original.reshape(batch_size, -1),
                reduction="none",
            ).sum(dim=-1)

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

        return (recon_loss + beta * KLD).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)

    def _cr_loss(self, mu, logvar, mu_aug, logvar_aug, gamma):
        """
        distance between two gaussians
        """
        std_orig = logvar.exp()
        std_aug = logvar_aug.exp()

        cr_loss = (
            0.5
            * torch.sum(
                2 * torch.log(std_orig / std_aug)
                - 1
                + (std_aug ** 2 + (mu_aug - mu) ** 2) / std_orig ** 2,
                dim=1,
            ).mean(dim=0)
        )

        cr_loss *= gamma
        return cr_loss

    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        # Sample N(0, I)
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    @classmethod
    def _load_model_config_from_folder(cls, dir_path):
        file_list = os.listdir(dir_path)

        if "model_config.json" not in file_list:
            raise FileNotFoundError(
                f"Missing model config file ('model_config.json') in"
                f"{dir_path}... Cannot perform model building."
            )

        path_to_model_config = os.path.join(dir_path, "model_config.json")
        model_config = CRVAEConfig.from_json_file(path_to_model_config)

        return model_config

    @classmethod
    def load_from_folder(cls, dir_path):
        """Class method to be used to load the model from a specific folder
        Args:
            dir_path (str): The path where the model should have been be saved.
        .. note::
            This function requires the folder to contain:
            - | a ``model_config.json`` and a ``model.pt`` if no custom architectures were provided
            **or**

            - | a ``model_config.json``, a ``model.pt`` and a ``encoder.pkl`` (resp.
                ``decoder.pkl``) if a custom encoder (resp. decoder) was provided
        """

        model_config = cls._load_model_config_from_folder(dir_path)
        model_weights = cls._load_model_weights_from_folder(dir_path)

        if not model_config.uses_default_encoder:
            encoder = cls._load_custom_encoder_from_folder(dir_path)

        else:
            encoder = None

        if not model_config.uses_default_decoder:
            decoder = cls._load_custom_decoder_from_folder(dir_path)

        else:
            decoder = None

        model = cls(model_config, encoder=encoder, decoder=decoder)
        model.load_state_dict(model_weights)

        return model
