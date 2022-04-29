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
from pythae.data.datasets import FolderDataset
import torchvision
import os
import glob


def generate_temp_datalist(root: str) -> List[str]:
    extensions = ["jpg", "jpeg", "png"]
    filenames = []
    for ext in extensions:
        filenames.extend(glob.glob(f"{root}/**/*.{ext}", recursive=True))
    return filenames


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def download_celeba(data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # download celeba via gdown
    celeba_url = "https://drive.google.com/u/0/uc?id=1ZRbdUMfeUMjrqGBbFgJlUIRRXqV7TeFF"
    os.system(f"gdown -O {os.path.join(data_dir, 'img_align_celeba.zip')} {celeba_url}")

    # unzip img_align_celeba.zip
    os.system(f"unzip {os.path.join(data_dir,'img_align_celeba.zip')} -d {data_dir}")

    os.system(f"rm {os.path.join(data_dir,'img_align_celeba.zip')}")

    assert os.path.exists(os.path.join(data_dir, "img_align_celeba"))
    assert len(os.listdir(os.path.join(data_dir, "img_align_celeba"))) > 0


def get_dataset(dataset_name: str, config) -> Tuple[Union[torch.Tensor, np.ndarray]]:
    """
    Get the dataset from the name.
    """
    train_percentage = 0.80

    if dataset_name == "CIFAR10":
        dataset = datasets.CIFAR10(root=config["data_dir"], train=True, download=True,)
    elif dataset_name == "CIFAR100":
        dataset = datasets.CIFAR100(root=config["data_dir"], train=True, download=True,)
    elif dataset_name == "MNIST":
        dataset = datasets.MNIST(root=config["data_dir"], train=True, download=True,)
    elif dataset_name == "CelebA":
        celeba_dir = os.path.join(config["data_dir"], "img_align_celeba")
        if not os.path.exists(celeba_dir) or len(os.listdir(celeba_dir)) < 200000:
            download_celeba(config["data_dir"])
        all_filenames = generate_temp_datalist(root=config["data_dir"])
        train_filenames = all_filenames[: int(train_percentage * len(all_filenames))]
        test_filenames = all_filenames[int(train_percentage * len(all_filenames)) :]
        train_data = FolderDataset("", train_filenames, (64, 64))
        test_data = FolderDataset("", test_filenames, (64, 64))
        return train_data, test_data

    else:
        raise ValueError(f"The dataset {dataset_name} is not implemented.")

    data = dataset.data
    del dataset

    data = torch.tensor(data).to(device)

    # Correct shape of single channel images
    if len(data.shape) == 3:
        data = data.unsqueeze(1)

    # make sure all images are in the range [0, 1]
    if data.max() > 1.0:
        data = data / 255.0

    # set the channel dimension to be dimension 1
    if data.shape[-1] in [1, 3]:
        data = data.permute(0, 3, 1, 2)

    train_data = data[: int(train_percentage * len(data))]
    test_data = data[int(train_percentage * len(data)) :]

    assert len(data.shape) == 4, f"The shape of the data is {data.shape}"
    assert data.shape[1] == 3, f"The number of channels is {data.shape[1]}"
    assert data.max() <= 1.0, f"the max value is {data.max()}"
    assert data.min() >= 0.0, f"the min value is {data.min()}"

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
        encoder = benchmarks.celeba.Encoder_VAE_CELEBA(ae_config)
        decoder = benchmarks.celeba.Decoder_AE_CELEBA(ae_config)

    return encoder, decoder


def get_model(
    model_name: str, input_dim: Tuple[int], encoder, decoder, config
) -> models.VAE:
    """
    Get the model from the name.
    """

    if model_name == "VAE":
        model_config = models.VAEConfig(input_dim=input_dim, **config["model_config"],)

        model = models.VAE(model_config, encoder, decoder)

    elif model_name == "BetaVAE":
        model_config = models.BetaVAEConfig(
            input_dim=input_dim, **config["model_config"],
        )

        model = models.BetaVAE(model_config, encoder, decoder)

    elif model_name == "DVAE":
        model_config = models.DVAEConfig(input_dim=input_dim, **config["model_config"],)
        model = models.DVAE(model_config, encoder, decoder)

    elif model_name == "DVAEAug":
        model_config = models.DVAEAugConfig(input_dim=input_dim, **config["model_config"],)
        model = models.DVAEAug(model_config, encoder, decoder)

    elif model_name == "CRVAE":
        model_config = models.CRVAEConfig(
            input_dim=input_dim, **config["model_config"],
        )

        model = models.CRVAE(model_config, encoder, decoder)
    elif model_name == "PVAE":
        model_config = models.PVAEConfig(
                input_dim=input_dim, **config["model_config"],
        )
        model = models.PVAE(model_config, encoder, decoder)
    return model


def get_pipeline(trainer_name: str, model: models.VAE, config) -> trainers.BaseTrainer:
    if trainer_name == "BaseTrainer":
        training_config = trainers.BaseTrainingConfig(**config["training_config"])

    else:
        raise ValueError(f"The trainer {trainer_name} is not implemented.")

    pipeline = pipelines.TrainingPipeline(model=model, training_config=training_config,)

    return pipeline


def correct_output_dir(config):
    output_folder = f"{config['dataset_name']}_{config['model_name']}"
    if output_folder not in config["training_config"]["output_dir"]:
        output_folder = os.path.join(
            config["training_config"]["output_dir"], output_folder
        )
        config["training_config"]["output_dir"] = output_folder
    return config


def train(config):
    """
    Main function.
    """
    # add dataset name and model name to the output folder
    config = correct_output_dir(config)

    # get the dataset
    train_data, test_data = get_dataset(config["dataset_name"], config)

    # get the encoder and decoder
    (encoder, decoder) = get_encoder_decoder(
        config["dataset_name"], train_data.shape[1:], config
    )

    # get the model
    model = get_model(
        config["model_name"], train_data.shape[1:], encoder, decoder, config,
    )

    # get the pipeline
    pipeline = get_pipeline(config["trainer_name"], model, config)

    # train the model
    pipeline(train_data, test_data)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, "r"))

    train(config)
