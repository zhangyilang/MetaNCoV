#!/usr/bin/bash

# Environment for miniImageNet & tieredImageNet
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install h5py ordered-set tqdm

# Commands in the following setup the environment for experiments on CUB dataset, which requires the original unmodified
# torchmeta package with pyTorch<=1.9.1, torchvision<=0.10.1 and cudatoolkit<=11.2
#conda install -c conda-forge pytorch-gpu=1.9.1 torchvision=0.10.1 torchaudio==0.9.1 cudatoolkit=11.2
#pip install torchmeta
