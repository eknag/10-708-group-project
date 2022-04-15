# Cloning and pulling

when cloning, make sure to run
```
git clone --recurse-submodules
```

When updating, run
```
git submodule update --remote
```

https://git-scm.com/book/en/v2/Git-Tools-Submodules

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