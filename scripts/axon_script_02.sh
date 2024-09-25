#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

## Non-IQN models
for LATENT_SIZE in {8..24}
do
    for JOB_IDX in {0..7}
    do
        python 02_run_gridworld_frozentransfer.py $LATENT_SIZE --job_idx $JOB_IDX --n_jobs 8 &
    done
    wait
done

# IQN models
for LATENT_SIZE in {8..24}
do
    for JOB_IDX in {0..7}
    do
        python 02_run_gridworld_frozentransfer.py $LATENT_SIZE --job_idx $JOB_IDX --n_jobs 8 --discount_factor 0.7 --iqn &
    done
    wait
done
