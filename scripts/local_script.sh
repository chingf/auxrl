#!/bin/bash 

source ~/.bashrc
conda activate sr
python 09_run_gridworld_cifar.py -1 1 dm 12 1.0
python 09_run_gridworld_cifar.py -1 1 dm 16 1.0
python 09_run_gridworld_cifar.py -1 1 dm 10 1.0

wait
