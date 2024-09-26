#!/usr/bin/bash

# miniImageNet + 4-block CNN
nohup python main.py --dataset miniImageNet --num-way 5 --num-supp 1 --algorithm MetaNCoVSGD --syl-dim 10 --task-lr 0.01 \
  > log/miniimagenet/MetaNCoV+MetaSGD-5way1shot.log 2>&1 &

nohup python main.py --dataset miniImageNet --num-way 5 --num-supp 5 --algorithm MetaNCoVSGD --syl-dim 5 --task-lr 0.01 \
  > log/miniimagenet/MetaNCoV+MetaSGD-5way5shot.log 2>&1 &


# Attention: before running the tests with WRN-28-10 embeddings, use the following commands to make the downloaded LEO
# datasets (which has been preprocessed and encoded with python 2) compatible with python 3.
# performed when transforming the dataset.
python ./src/leo_embeddings.py

# miniImageNet + WRN-28-10 (use --crop center/multiview to test with different embedding crops)
nohup python main.py --dataset LEO-miniImageNet --crop multiview --num-way 5 --num-supp 1 --algorithm MetaNCoVSGD --syl-dim 10 --task-lr 2 \
  > log/leo-miniimagenet/MetaNCoV+MC-multiview-5way1shot.log 2>&1 &

nohup python main.py --dataset LEO-miniImageNet --crop multiview --num-way 5 --num-supp 5 --algorithm MetaNCoVSGD --syl-dim 10 --task-lr 2 \
  > log/leo-miniimagenet/MetaNCoV+MC-multiview-5way5shot.log 2>&1 &

# tieredImageNet + WRN-28-10 (center crop only)
nohup python main.py --dataset LEO-tieredImageNet --crop center --num-way 5 --num-supp 1 --algorithm MetaNCoVSGD --syl-dim 10 --task-lr 2 \
  > log/leo-tieredimagenet/MetaNCoV+MC-multiview-5way1shot.log 2>&1 &


# CUB + 4-block CNN (CUB dataset requires downgrading the packages; see env_setup.sh)
nohup python main.py --dataset CUB --num-way 5 --num-supp 1 --algorithm MetaNCoVSGD --syl-dim 5 --task-lr 0.01 \
  > log/cub/MetaNCoV+MetaSGD-5way1shot.log 2>&1 &

nohup python main.py --dataset CUB num-way 5 num-supp 5 --algorithm MetaNCoVSGD --syl-dim 5 --task-lr 0.01 \
  > log/cub/MetaNCoV+MetaSGD-5way5shot.log 2>&1 &
