#!/bin/bash

# install.sh

# --
# Setup environment

conda create -y -n sod_env python=3.7
conda activate sod_env

conda install -y -c pytorch pytorch torchvision cudatoolkit=10.1

pip install matplotlib
pip install tqdm
# pip install git+https://github.com/bkj/rcode