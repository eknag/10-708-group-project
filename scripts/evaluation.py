from __future__ import annotations
from argparse import ArgumentParser
from pythae.models import *
from pythae.models.crvae.crvae_model import CRVAE
from pythae.samplers import *
from torchvision import datasets
from torchvision.datasets.vision import VisionDataset
import torchvision
from pytorch_fid.fid_score import calculate_fid_given_paths
import os
from tqdm import tqdm
from functools import partial
import torch
import yaml
import dotsi
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

from lipschitz.lipschitz_calc import get_lipschitz

from torch.utils.data import DataLoader
from nngeometry.generator import Jacobian
from nngeometry.object import PMatDiag
from nngeometry.metrics import FIM
from torchvision import transforms


ENCODER_NAME = "_encoder"
DECODER_NAME = "_decoder"


def get_model(model_name: str) -> VAE:
    if model_name == "VAE":
        return VAE
    elif model_name == "BetaVAE":
        return BetaVAE
    elif model_name == "DVAE":
        return DVAE
    elif model_name == "CRVAE":
        return CRVAE


def get_sampler(sampler_name: str) -> BaseSampler:
    if sampler_name == "NormalSampler":
        return NormalSampler
    elif sampler_name == "GaussianMixtureSampler":
        return GaussianMixtureSampler


def get_eval_dataset(dataset_name: str, dataset_dir) -> VisionDataset:
    if dataset_name == "MNIST":
        return datasets.MNIST(root=dataset_dir, train=False, download=True)
    elif dataset_name == "FashionMNIST":
        return datasets.FashionMNIST(root=dataset_dir, train=False, download=True)
    elif dataset_name == "CIFAR10":
        return datasets.CIFAR10(root=dataset_dir, train=False, download=True)
    elif dataset_name == "CELEBA":
        return datasets.CelebA(root=dataset_dir, train=False, download=True)


def get_newest_file(path: str) -> str:
    files = os.listdir(path)
    files.sort(key=lambda x: os.path.getmtime(os.path.join(path, x)))
    return os.path.join(path, files[-1])


def evaluate(
    model_name: str,
    dataset_name: str,
    sampler_name: str,
    num_samples: int,
    num_workers: int,
    batch_size: int,
    output_dir: str,
    model_dir: str,
    dataset_dir: str,
    calc_sing_val: bool,
    lipschitz: bool,
    curve_est: bool,
) -> dict[str, float]:

    MODEL = get_model(model_name)
    model_dir = os.path.join(model_dir, f"{dataset_name}_{model_name}")
    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"Model directory {model_dir} not found, make sure to train the model first."
        )

    model_file = os.path.join(get_newest_file(model_dir), "final_model")
    model: VAE = MODEL.load_from_folder(model_file)
    model.eval()

    if calc_sing_val:
        # Calculate singular values for encoder and decoder networks
        spectral, lip = calc_sing_val(model.encoder, output_dir, dataset_name +  "_" + model_name + ENCODER_NAME)
        spectral, lip = calc_sing_val(model.decoder, output_dir, dataset_name +  "_" + model_name + DECODER_NAME)
    if lipschitz:
        # Calculate Lipschitz constants for encoder and decoder networks.  Note: this reads singular value files
        # from output_dir and stores Lipschitz constants in output_dir/LIP_OUT_SUBDIR (defined in lipschitz_calc.py)
        spectral, lip = get_lipschitz(model.encoder, output_dir, dataset_name +  "_" + model_name + ENCODER_NAME)
        print("Encoder network lipschitz constant: ", lip, " (spectral norm: " ,  spectral,  ")")
        spectral, lip = get_lipschitz(model.decoder, output_dir, dataset_name +  "_" + model_name + DECODER_NAME)
        print("Decoder network lipschitz constant: ", lip, " (spectral norm: " ,  spectral,  ")")
    if curve_est:
        # Curvature estimation
        def get_dataset(dataset_name: str, dataset_dir) -> VisionDataset:
            if dataset_name == "MNIST":
                return datasets.MNIST(root=dataset_dir, train=True, download=True)
            elif dataset_name == "FashionMNIST":
                return datasets.FashionMNIST(root=dataset_dir, train=True, download=True)
            elif dataset_name == "CIFAR10":
                return datasets.CIFAR10(root=dataset_dir, train=True, download=True)
            elif dataset_name == "CELEBA":
                return datasets.CelebA(root=dataset_dir, train=True, download=True)
        in_dataset = get_dataset(dataset_name, dataset_dir)
        convert_tensor = transforms.ToTensor()
        dataset = []
        for d in in_dataset:
            data = convert_tensor(d[0])
            dataset.append((data.view(1, *data.shape), d[1]))

        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        def encoder_function(*dataset):
            # Run all samples in dataset through the model and return a Tensor of their embeddings
            out = []
            for i in range(len(dataset[0])):
                out.append(model.encoder(dataset[0][i])['embedding'])
            return torch.stack(out, dim=0)

        K = FIM(model=model.encoder, loader=loader, representation=PMatDiag, n_output=16, function=encoder_function)   
        print(model_name, " (", dataset_name, ") Encoder: Fischer Inf. Mx: ", K)
        return

    SAMPLER = get_sampler(sampler_name)
    sampler = SAMPLER(model=model)

    # make the output directory
    sample_output_dir = os.path.join(output_dir, model_name, sampler_name, dataset_name)
    os.makedirs(sample_output_dir, exist_ok=True)

    samples = sampler.sample(num_samples)
    assert (
        len(samples) == num_samples
    ), f"Expected {num_samples} samples, got {len(samples)}"

    for i, sample in tqdm(enumerate(samples)):
        sample = sample.cpu().squeeze(0)
        torchvision.utils.save_image(
            sample, os.path.join(sample_output_dir, f"{i}.png")
        )

    dataset = get_eval_dataset(dataset_name, dataset_dir)
    assert (
        len(dataset) >= num_samples
    ), f"Dataset {dataset_name} has less than {num_samples} samples"

    data = dataset.data

    eval_output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(eval_output_dir, exist_ok=True)
    assert os.path.exists(eval_output_dir), f"Could not find {eval_output_dir}"

    for i in range(num_samples):
        image_output_dir = os.path.join(eval_output_dir, f"{i}.png")
        img = data[i]
        if isinstance(img, torch.Tensor):
            img = img.cpu().squeeze(0)
            torchvision.utils.save_image(img, image_output_dir)
        elif isinstance(img, np.ndarray):
            img = img.squeeze()
            img = Image.fromarray(img)
            img.save(image_output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"batch size {batch_size}")
    fid = float(
        calculate_fid_given_paths(
            (eval_output_dir, sample_output_dir),
            batch_size,
            device,
            dims=2048,
            num_workers=num_workers,
        )
    )

    results = {"FID": fid}
    print(f"FID: {fid}")
    # remove directories and files
    os.system(f"rm -rf {eval_output_dir}")
    os.system(f"rm -rf {sample_output_dir}")
    return results


def main():
    sampler_name = config.sampler
    num_samples = config.num_samples
    num_workers = config.num_workers
    batch_size = config.batch_size
    output_dir = config.output_dir
    model_dir = config.model_dir
    dataset_dir = config.dataset_dir
    calc_sing_val = config.calc_sing_val
    lipschitz = config.lipschitz
    curve_est = config.curve_est
    performances = {}
    for dataset_name in config.datasets:
        dataset_performance = {}
        for model_name in config.models:
            results = evaluate(
                model_name,
                dataset_name,
                sampler_name,
                num_samples,
                num_workers,
                batch_size,
                output_dir,
                model_dir,
                dataset_dir,
                calc_sing_val,
                lipschitz,
                curve_est
            )
            dataset_performance[model_name] = results
        performances[dataset_name] = dataset_performance

    # save results
    results_file = os.path.join(output_dir, "evaluation_results.yaml")
    with open(results_file, "w") as f:
        yaml.dump(performances, f, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    # add config file argument with default as configs/eval.yaml
    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        default="configs/eval.yaml",
        help="config file in yaml",
    )
    args = parser.parse_args()

    assert args.config is not None, "config file is required"
    config = yaml.safe_load(open(args.config, "r"))
    config = dotsi.Dict(config)

    main()

