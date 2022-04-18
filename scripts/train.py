from __future__ import annotations

import argparse
from typing import Tuple, Union

import numpy as np
import pythae.models as models
import pythae.models.nn.benchmarks as benchmarks
import pythae.pipelines as pipelines
import pythae.trainers as trainers
import torch
import yaml
from torchvision import datasets


def get_dataset(dataset_name: str, config) -> Tuple[Union[torch.Tensor, np.ndarray]]:
    """
    Get the dataset from the name.
    """
    train_percentage = 0.80

    if dataset_name == "CIFAR10":
        dataset = datasets.CIFAR10(
            root=config["data_dir"],
            train=True,
            download=True,
        )
        data = dataset.data
        data = torch.tensor(data).to(device)
        data = data.permute(0, 3, 1, 2)
        del dataset

    train_data = data[: int(train_percentage * len(data))]
    test_data = data[int(train_percentage * len(data)) :]

    assert len(data.shape) == 4, f"The shape of the data is {data.shape}"
    assert data.shape[1] == 3, f"The number of channels is {data.shape[1]}"

    # # need to check data format
    # assert data.max() <= 1.0
    # assert data.min() >= 0.0
    return train_data, test_data


def get_encoder_decoder(dataset_name: str, input_dim: Tuple[int], config):
    ae_config = models.base.base_config.BaseAEConfig(
        input_dim=input_dim,
        latent_dim=config["model_config"]["latent_dim"],
        uses_default_encoder=False,
        uses_default_decoder=False,
    )

    if "CIFAR" in dataset_name:
        encoder = benchmarks.cifar.Encoder_VAE_CIFAR(ae_config)
        decoder = benchmarks.cifar.Decoder_AE_CIFAR(ae_config)

    elif "MNIST" in dataset_name:
        encoder = benchmarks.mnist.Encoder_VAE_MNIST(ae_config)
        decoder = benchmarks.mnist.Decoder_AE_MNIST(ae_config)

    elif "CelebA" in dataset_name:
        encoder = benchmarks.celeba.Encoder_VAE_CelebA(ae_config)
        decoder = benchmarks.celeba.Decoder_AE_CelebA(ae_config)

    return encoder, decoder


def get_model(
    model_name: str, input_dim: Tuple[int], encoder, decoder, config
) -> models.VAE:
    """
    Get the model from the name.
    """

    if model_name == "VAE":
        model_config = models.VAEConfig(
            input_dim=input_dim,
            **config["model_config"],
        )

        model = models.VAE(model_config, encoder, decoder)

    elif model_name == "BetaVAE":
        model_config = models.BetaVAEConfig(
            input_dim=input_dim,
            **config["model_config"],
        )

        model = models.BetaVAE(model_config, encoder, decoder)

    elif model_name == "DVAE":
        model_config = models.DVAEConfig(
            input_dim=input_dim,
            **config["model_config"],
        )
        model = models.DVAE(model_config, encoder, decoder)

    elif model_name == "CRVAE":
        model_config = models.CRVAEConfig(
            input_dim=input_dim,
            **config["model_config"],
        )

        model = models.CRVAE(model_config, encoder, decoder)

    return model


def get_pipeline(trainer_name: str, model: models.VAE, config) -> trainers.BaseTrainer:
    if trainer_name == "BaseTrainer":
        training_config = trainers.BaseTrainingConfig(**config["training_config"])

    else:
        raise ValueError(f"The trainer {trainer_name} is not implemented.")

    print(f"The trainer is {trainer_name}")
    print(f"the learning rate is {training_config.learning_rate}")

    pipeline = pipelines.TrainingPipeline(
        model=model,
        training_config=training_config,
    )

    return pipeline


def main(config):
    """
    Main function.
    """
    # get the dataset
    train_data, test_data = get_dataset(config["dataset_name"], config)

    # get the encoder and decoder
    (encoder, decoder) = get_encoder_decoder(
        config["dataset_name"], train_data.shape[1:], config
    )

    # get the model
    model = get_model(
        config["model_name"],
        train_data.shape[1:],
        encoder,
        decoder,
        config,
    )

    # get the pipeline
    pipeline = get_pipeline(config["trainer_name"], model, config)

    # train the model
    pipeline(train_data, test_data)


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, "r"))

    main(config)
