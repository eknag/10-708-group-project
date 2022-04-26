#!/bin/bash

data_dir=./aws_data
s3_bucket=s3://10-708-group-project/

aws s3 sync $s3_bucket $data_dir
