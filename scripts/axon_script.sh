#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

# Runs basic gridworld experiments

python 01_run_gridworld.py 10 basic_test --job_idx 0 --n_jobs 2 &
python 01_run_gridworld.py 10 basic_test --job_idx 1 --n_jobs 2 &

wait
