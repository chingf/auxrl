#!/bin/bash 

# Last run was 14-21

python 01_run_gridworld.py -1 1 dm_large_q 17 0.9 shuffle
python 02_run_gridworld_frozentransfer.py -1 1 dm_large_q 17 0.9 shuffle

python 01_run_gridworld.py -1 1 dm_large_q 17 0.8 shuffle
python 02_run_gridworld_frozentransfer.py -1 1 dm_large_q 17 0.8 shuffle

python 01_run_gridworld.py -1 1 dm_large_q 17 0.7 shuffle
python 02_run_gridworld_frozentransfer.py -1 1 dm_large_q 17 0.7 shuffle

python 01_run_gridworld.py -1 1 dm_large_q 17 0.6 shuffle
python 02_run_gridworld_frozentransfer.py -1 1 dm_large_q 17 0.6 shuffle

python 01_run_gridworld.py -1 1 dm_large_q 17 0.5 shuffle
python 02_run_gridworld_frozentransfer.py -1 1 dm_large_q 17 0.5 shuffle

python 01_run_gridworld.py -1 1 dm_large_q 17 0.4 shuffle
python 02_run_gridworld_frozentransfer.py -1 1 dm_large_q 17 0.4 shuffle

python 01_run_gridworld.py -1 1 dm_large_q 17 0.3 shuffle
python 02_run_gridworld_frozentransfer.py -1 1 dm_large_q 17 0.3 shuffle

python 01_run_gridworld.py -1 1 dm_large_q 17 0.2 shuffle
python 02_run_gridworld_frozentransfer.py -1 1 dm_large_q 17 0.2 shuffle


wait
