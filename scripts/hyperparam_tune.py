from __future__ import annotations
import argparse
import yaml
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt import hp
from train import train

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", type=str, default="configs/train_crvae.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.base_config, "r"))

    lr = hp.loguniform("lr", 1e-6, 1e-2)
    latent_dim = hp.choice("latent_dim", [16, 50, 256])
    gamma = hp.choice("gamma", [0.1, 0.001])

    config["training_config"]["learning_rate"] = lr
    config["model_config"]["latent_dim"] = latent_dim
    config["model_config"]["gamma"] = gamma

    hyperopt_search = HyperOptSearch(config, metric="eval_loss", mode="min")

    analysis = tune.run(
        train,
        num_samples=1,
        search_alg=hyperopt_search,
        resources_per_trial={"gpu": 1},
        raise_on_failed_trial=False,
    )
