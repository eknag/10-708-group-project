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

    lr = tune.loguniform(1e-6, 1e-2)
    batch_size = tune.choice([64, 256])
    # gamma = hp.choice("gamma", [0.1, 0.001])

    config["training_config"]["learning_rate"] = lr
    config["training_config"]["batch_size"] = batch_size

    ray.init(log_to_driver=False)

    analysis = tune.run(
        train,
        config=config,
        num_samples=10,
        resources_per_trial={"gpu": 1, "cpu": 8},
        raise_on_failed_trial=False,
        scheduler=ASHAScheduler(metric="eval_loss", mode="min"),
        local_dir="./ray_results",
        name="crvae_simple_training_params",
    )
