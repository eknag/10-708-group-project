#!/bin/bash

data_dir=./results
s3_bucket=s3://10-708-group-project/trained_models/

aws s3 sync $data_dir $s3_bucket
