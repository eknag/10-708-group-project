#!/bin/bash

config_dir=configs
models=("vae" "betavae" "dvae")

for i in ${!models[@]}; do
    echo "Training $models[$i]"
    python scripts/train.py --config "${config_dir}/train_${models[$i]}.yaml"
done