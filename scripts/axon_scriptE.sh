#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT


# from script C
python 01_run_gridworld.py 10 full_gridsearch --job_idx 0 --n_jobs 2 --discount_factor 0.9 --shuffle &
python 01_run_gridworld.py 10 full_gridsearch --job_idx 1 --n_jobs 2 --discount_factor 0.9 --shuffle &
wait


# from script A

# Latent Size 10, shuffled observations observations
python 01_run_gridworld.py 10 full_gridsearch --job_idx 0 --n_jobs 2 --discount_factor 0.5 --shuffle &
python 01_run_gridworld.py 10 full_gridsearch --job_idx 1 --n_jobs 2 --discount_factor 0.5 --shuffle &
wait

