from __future__ import annotations
import yaml
import argparse
from train import train
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_crvae.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, "r"))

    output_dir = config["training_config"]["output_dir"]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for latent_dim in [2, 4, 16, 50, 128, 256, 1024, 2048]:
        config["model_config"]["latent_dim"] = latent_dim
        config["training_config"]["output_dir"] = output_dir
        train(config)
