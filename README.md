# Setup
`source setup.sh`
`cd benchmark_VAE`
`pip install -e .`


# AWS
When training or evaluating models, you need to make sure you have awscli installed and configured to access s3://10-708-group-project/

# Downloading and Uploading Trained Models
**BEFORE RUNNING** Change the data_dir

To download trained models, run `bash download.sh` 

To upload trained models, run `bash upload.sh`

# Training
Run `python3 scripts/train.py --config configs/<training config file>` in order to train a model.

Run `python3 scripts/evaluation.py -c configs/eval.yaml` to perform evaluations.

