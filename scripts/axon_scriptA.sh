#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

# Runs basic gridworld experiments

# Latent Size 7, regular observations
python 01_run_gridworld.py 7 full_gridsearch --job_idx 0 --n_jobs 2 --discount_factor 0.5 &
python 01_run_gridworld.py 7 full_gridsearch --job_idx 1 --n_jobs 2 --discount_factor 0.5 &
wait

# Latent Size 7, shuffled observations
python 01_run_gridworld.py 7 full_gridsearch --job_idx 0 --n_jobs 2 --discount_factor 0.5 --shuffle &
python 01_run_gridworld.py 7 full_gridsearch --job_idx 1 --n_jobs 2 --discount_factor 0.5 --shuffle &
wait



