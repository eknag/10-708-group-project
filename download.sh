#!/bin/bash

data_dir=~/data/10-708-project-results/trained_models
s3_bucket=s3://10-708-group-project/trained_models/

aws s3 sync $s3_bucket $data_dir
