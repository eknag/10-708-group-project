#!/bin/bash

data_dir=./results
s3_bucket=s3://10-708-group-project/crvae_latent_dim_test/

aws s3 sync $data_dir $s3_bucket