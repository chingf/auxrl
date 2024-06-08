#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

## Unshuffled Gridworld

# Non-IQN models
#for LATENT_SIZE in {3..12}
#do
#    for JOB_IDX in {0..15}
#    do
#        python 01_run_gridworld.py $LATENT_SIZE selected_models_gridworld --job_idx $JOB_IDX --n_jobs 16 &
#    done
#    wait
#done

# IQN models
#for LATENT_SIZE in {3..12}
#do
#    for JOB_IDX in {0..15}
#    do
#        python 01_run_gridworld.py $LATENT_SIZE mf0 --job_idx $JOB_IDX --n_jobs 16 --discount_factor 0.7 --iqn &
#    done
#    wait
#done

## Shuffled Gridworld

# Non-IQN models
for LATENT_SIZE in {3..24}
do
    for JOB_IDX in {0..15}
    do
        python 01_run_gridworld.py $LATENT_SIZE selected_models_grid_shuffle --job_idx $JOB_IDX --n_jobs 16 --shuffle &
    done
    wait
done

# IQN models
for LATENT_SIZE in {3..24}
do
    for JOB_IDX in {0..15}
    do
        python 01_run_gridworld.py $LATENT_SIZE mf1 --job_idx $JOB_IDX --n_jobs 16 --discount_factor 0.7 --iqn --shuffle &
    done
    wait
done
