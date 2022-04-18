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
        dataset = datasets.CIFAR10(root=config.data_dir, split="train", download=True)
        data = dataset.data
        del dataset

    train_data = data[: int(train_percentage * len(data))]
    test_data = data[int(train_percentage * len(data)) :]

    assert len(data.shape) == 4 and data.shape[1] == 3

    # need to check data format
    assert data.max() <= 1.0
    assert data.min() >= 0.0
    return train_data, test_data


def get_model(model_name: str, input_dim: Tuple[int], config) -> models.VAE:
    """
    Get the model from the name.
    """

    if model_name == "VAE":
        model_config = models.VAEConfig(
            input_dim=input_dim,
            latent_dim=config.model_config.latent_dim,
        )

        encoder = benchmarks.Encoder_VAE_GENERIC(model_config)
        decoder = benchmarks.Decoder_AE_GENERIC(model_config)

        model = models.VAE(encoder, decoder, model_config)

    elif model_name == "BetaVAE":
        model_config = models.BetaVAEConfig(
            input_dim=input_dim,
            latent_dim=config.model_config.latent_dim,
            beta=config.model_config.beta,
        )

        encoder = benchmarks.Encoder_VAE_GENERIC(model_config)
        decoder = benchmarks.Decoder_AE_GENERIC(model_config)

        model = models.BetaVAE(encoder, decoder, model_config)

    elif model_name == "DVAE":
        model_config = models.DVAEConfig(
            input_dim=input_dim,
            latent_dim=config.model_config.latent_dim,
            sigma=config.model_config.sigma,
        )

        encoder = benchmarks.Encoder_VAE_GENERIC(model_config)
        decoder = benchmarks.Decoder_AE_GENERIC(model_config)

        model = models.DVAE(encoder, decoder, model_config)

    elif model_name == "CRVAE":
        model_config = models.CRVAEConfig(
            input_dim=input_dim,
            # **config.model_config.__dict__,
            latent_dim=config.model_config.latent_dim,
            transform=config.model_config.transform,
            gamma=config.model_config.gamma,  # default 1e-3
            beta_1=config.model_config.beta_1,  # default 1
            beta_2=config.model_config.beta_2,  # default 1
        )

        encoder = benchmarks.Encoder_VAE_GENERIC(model_config)
        decoder = benchmarks.Decoder_AE_GENERIC(model_config)

        model = models.CRVAE(encoder, decoder, model_config)

    return model


def get_pipeline(trainer_name: str, model: models.VAE, config) -> trainers.BaseTrainer:
    if trainer_name == "BaseTrainer":
        training_config = trainers.BaseTrainingConfig(**config.training_config.__dict__)

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
    train_data, test_data = get_dataset(config.dataset_name, config)

    # get the model
    model = get_model(config.model_name, train_data.shape[1:], config)

    # get the pipeline
    pipeline = get_pipeline(config.trainer_name, model, config)

    # train the model
    pipeline.train(train_data, test_data)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, "r"))
    main(config)
