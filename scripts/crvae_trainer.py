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

    for dataset_name in ["CelebA"]:
        for aug_type in [
            "simple",
            "large",
            "simple_vertical_flip",
            "large_vertical_flip",
            "simple_jitter",
            "large_jitter",
            "simple_vertical_flip_jitter",
            "large_vertical_flip_jitter",
            "random_noise",
            "denoise",
            "ins_filter",
            "change_aspect_ratio",
        ]:
            config["model_config"]["aug_type"] = aug_type
            config["dataset_name"] = dataset_name
            config["training_config"]["output_dir"] = output_dir
            train(config)
