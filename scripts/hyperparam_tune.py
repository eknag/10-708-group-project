from __future__ import annotations
import argparse
import yaml
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt import hp
from train import train
import ray

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", type=str, default="configs/train_crvae.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.base_config, "r"))

    latent_dim = tune.grid_search([2, 16, 50, 128, 512, 1024, 4096])

    config["model_config"]["latent_dim"] = latent_dim

    ray.init(log_to_driver=False)

    analysis = tune.run(
        train,
        config=config,
        num_samples=1,
        resources_per_trial={"gpu": 1, "cpu": 8},
        raise_on_failed_trial=False,
        scheduler=ASHAScheduler(metric="eval_loss", mode="min"),
        local_dir="./ray_results",
        name="latent_dim_test",
    )
