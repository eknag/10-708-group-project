from __future__ import annotations
from argparse import ArgumentParser
from pythae.models import *
from pythae.samplers import *
from pythae.models import DVAE
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
    
def get_model(model_name: str) ->  VAE:
    if model_name == 'VAE':
        return VAE
    elif model_name == 'BetaVAE':
        return BetaVAE
    elif model_name == 'DVAE':
        return DVAE
    elif model_name == 'CRVAE':
        return CRVAE

def get_sampler(sampler_name: str) -> BaseSampler:
    if sampler_name == 'NormalSampler':
        return NormalSampler
    elif sampler_name == 'GaussianMixtureSampler':
        return GaussianMixtureSampler

def get_eval_dataset(dataset_name: str) -> VisionDataset:
    if dataset_name == 'MNIST':
        return datasets.MNIST(root=config.dataset_dir, train=False, download=True)
    elif dataset_name == 'FashionMNIST':
        return datasets.FashionMNIST(root=config.dataset_dir, train=False, download=True)
    elif dataset_name == 'CIFAR10':
        return datasets.CIFAR10(root=config.dataset_dir, train=False, download=True)
    elif dataset_name == 'CELEBA':
        return datasets.CelebA(root=config.dataset_dir, train=False, download=True)



def get_newest_file(path: str) -> str:
    files = os.listdir(path)
    files.sort(key=lambda x: os.path.getmtime(os.path.join(path, x)))
    return os.path.join(path,files[-1])




def main():
    MODEL = get_model(config.model)
    model_dir = os.path.join(config.model_dir, f"{config.dataset_name}_{config.model}")
    model_file = os.path.join(get_newest_file(model_dir), 'final_model')
    model : VAE = MODEL.load_from_folder(model_file)
    model.eval()

    SAMPLER = get_sampler(config.sampler)
    sampler = SAMPLER(model = model)

    # make the output directory
    sample_output_dir = os.path.join(config.output_dir,  config.model, config.sampler, config.dataset_name)
    os.makedirs(sample_output_dir, exist_ok=True)


    samples = sampler.sample(config.num_samples)
    assert len(samples) == config.num_samples, f"Expected {config.num_samples} samples, got {len(samples)}"
    
    for i, sample in tqdm(enumerate(samples)):
        sample = sample.cpu().squeeze(0)
        torchvision.utils.save_image(sample, os.path.join(sample_output_dir, f'{i}.png'))

    dataset = get_eval_dataset(config.dataset_name)
    assert len(dataset) >= config.num_samples, f"Dataset {config.dataset_name} has less than {config.num_samples} samples"

    data = dataset.data

    eval_output_dir = os.path.join(config.output_dir, config.dataset_name)
    os.makedirs(eval_output_dir, exist_ok=True)
    assert os.path.exists(eval_output_dir), f"Could not find {eval_output_dir}"

    for i in range(config.num_samples):
        image_output_dir = os.path.join(eval_output_dir, f'{i}.png')
        img = data[i]
        if isinstance(img, torch.Tensor):
            img = img.cpu().squeeze(0)
            torchvision.utils.save_image(img, image_output_dir)
        elif isinstance(img, np.ndarray):
            img = img.squeeze()
            img = Image.fromarray(img)
            img.save(image_output_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"batch size {config.batch_size}")
    fid = calculate_fid_given_paths(
        (eval_output_dir, sample_output_dir),
        config.batch_size,
        device,
        dims = 2048,
        num_workers=config.num_workers,
    )
    print(f"FID: {fid}")
    # remove directories and files
    os.system(f"rm -rf {eval_output_dir}")
    os.system(f"rm -rf {sample_output_dir}")



        

        
    

if __name__== "__main__":
    parser = ArgumentParser()
    # add config file argument with default as configs/eval.yaml
    parser.add_argument("-c", "--config", dest="config", default="configs/eval.yaml", help="config file in yaml")
    args = parser.parse_args()

    assert args.config is not None, "config file is required"
    config = yaml.safe_load(open(args.config, 'r'))
    config = dotsi.Dict(config)
    
    main()



    