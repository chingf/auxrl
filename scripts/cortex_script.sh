#!/bin/bash 

python 01_run_gridworld.py -1 1 dm_large_encoder 2 shuffle
python 01_run_gridworld.py -1 1 dm_large_encoder 3 shuffle
python 01_run_gridworld.py -1 1 dm_large_encoder 4 shuffle
python 01_run_gridworld.py -1 1 dm_large_encoder 5 shuffle
python 01_run_gridworld.py -1 1 dm_large_encoder 6 shuffle
python 01_run_gridworld.py -1 1 dm_large_encoder 7 shuffle
python 01_run_gridworld.py -1 1 dm_large_encoder 8 shuffle
python 01_run_gridworld.py -1 1 dm_large_encoder 9 shuffle
python 01_run_gridworld.py -1 1 dm_large_encoder 10 shuffle
python 01_run_gridworld.py -1 1 dm_large_encoder 11 shuffle

python 01_run_gridworld.py -1 1 dm_large_q 2 shuffle
python 01_run_gridworld.py -1 1 dm_large_q 3 shuffle
python 01_run_gridworld.py -1 1 dm_large_q 4 shuffle
python 01_run_gridworld.py -1 1 dm_large_q 5 shuffle
python 01_run_gridworld.py -1 1 dm_large_q 6 shuffle
python 01_run_gridworld.py -1 1 dm_large_q 7 shuffle
python 01_run_gridworld.py -1 1 dm_large_q 8 shuffle
python 01_run_gridworld.py -1 1 dm_large_q 9 shuffle
python 01_run_gridworld.py -1 1 dm_large_q 10 shuffle
python 01_run_gridworld.py -1 1 dm_large_q 11 shuffle

python 02_run_gridworld_frozentransfer.py -1 1 dm_large_encoder 2 shuffle
python 02_run_gridworld_frozentransfer.py -1 1 dm_large_encoder 3 shuffle
python 02_run_gridworld_frozentransfer.py -1 1 dm_large_encoder 4 shuffle
python 02_run_gridworld_frozentransfer.py -1 1 dm_large_encoder 5 shuffle
python 02_run_gridworld_frozentransfer.py -1 1 dm_large_encoder 6 shuffle
python 02_run_gridworld_frozentransfer.py -1 1 dm_large_encoder 7 shuffle
python 02_run_gridworld_frozentransfer.py -1 1 dm_large_encoder 8 shuffle
python 02_run_gridworld_frozentransfer.py -1 1 dm_large_encoder 9 shuffle
python 02_run_gridworld_frozentransfer.py -1 1 dm_large_encoder 10 shuffle
python 02_run_gridworld_frozentransfer.py -1 1 dm_large_encoder 11 shuffle

python 02_run_gridworld_frozentransfer.py -1 1 dm_large_q 2 shuffle
python 02_run_gridworld_frozentransfer.py -1 1 dm_large_q 3 shuffle
python 02_run_gridworld_frozentransfer.py -1 1 dm_large_q 4 shuffle
python 02_run_gridworld_frozentransfer.py -1 1 dm_large_q 5 shuffle
python 02_run_gridworld_frozentransfer.py -1 1 dm_large_q 6 shuffle
python 02_run_gridworld_frozentransfer.py -1 1 dm_large_q 7 shuffle
python 02_run_gridworld_frozentransfer.py -1 1 dm_large_q 8 shuffle
python 02_run_gridworld_frozentransfer.py -1 1 dm_large_q 9 shuffle
python 02_run_gridworld_frozentransfer.py -1 1 dm_large_q 10 shuffle
python 02_run_gridworld_frozentransfer.py -1 1 dm_large_q 11 shuffle

wait
