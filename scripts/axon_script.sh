#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

## Unshuffled Gridworld

# Non-IQN models
#for LATENT_SIZE in {3..12}
#do
#    for JOB_IDX in {0..5}
#    do
#        python 01_run_gridworld.py $LATENT_SIZE selected_models_gridworld --job_idx $JOB_IDX --n_jobs 6 &
#    done
#    wait
#done

# IQN models
#for LATENT_SIZE in {3..12}
#do
#    for JOB_IDX in {0..5}
#    do
#        python 01_run_gridworld.py $LATENT_SIZE mf0 --job_idx $JOB_IDX --n_jobs 6 --discount_factor 0.7 --iqn &
#    done
#    wait
#done

## Shuffled Gridworld

# Non-IQN models
#for LATENT_SIZE in {3..24}
#do
#    for JOB_IDX in {0..5}
#    do
#        python 01_run_gridworld.py $LATENT_SIZE selected_models_grid_shuffle --job_idx $JOB_IDX --n_jobs 6 --shuffle &
#    done
#    wait
#done

## IQN models
#for LATENT_SIZE in {3..24}
#do
#    for JOB_IDX in {0..5}
#    do
#        python 01_run_gridworld.py $LATENT_SIZE mf1 --job_idx $JOB_IDX --n_jobs 6 --discount_factor 0.7 --iqn --shuffle &
#    done
#    wait
#done

## CIFAR Gridworld

# Non-IQN models
#for LATENT_SIZE in {3..11}
#do
#    for JOB_IDX in {0..5}
#    do
#        python 01_run_gridworld.py $LATENT_SIZE selected_models_grid_shuffle --job_idx $JOB_IDX --n_jobs 6 --cifar &
#    done
#    wait
#done

# IQN models
#for LATENT_SIZE in {3..11}
#do
#    for JOB_IDX in {0..5}
#    do
#        python 01_run_gridworld.py $LATENT_SIZE mf1 --job_idx $JOB_IDX --n_jobs 6 --discount_factor 0.7 --iqn --cifar &
#    done
#    wait
#done

## Shuffled Gridworld, Large Encoder

## Non-IQN models
#for LATENT_SIZE in {8..8}
#do
#    for JOB_IDX in {0..7}
#    do
#        python 01_run_gridworld.py $LATENT_SIZE full_gridsearch --nn_yaml dm_large_encoder --job_idx $JOB_IDX --n_jobs 8 --discount_factor 0.7 --shuffle &
#    done
#    wait
#done
#
#for LATENT_SIZE in {8..8}
#do
#    for JOB_IDX in {0..7}
#    do
#        python 01_run_gridworld.py $LATENT_SIZE full_gridsearch --nn_yaml dm_large_encoder --job_idx $JOB_IDX --n_jobs 8 --shuffle &
#    done
#    wait
#done
#
## IQN models
#for LATENT_SIZE in {8..8}
#do
#    for JOB_IDX in {0..7}
#    do
#        python 01_run_gridworld.py $LATENT_SIZE mf_gridsearch --nn_yaml dm_large_encoder --job_idx $JOB_IDX --n_jobs 8 --discount_factor 0.7 --iqn --shuffle &
#    done
#    wait
#done
#
#for LATENT_SIZE in {8..8}
#do
#    for JOB_IDX in {0..7}
#    do
#        python 01_run_gridworld.py $LATENT_SIZE mf_gridsearch --nn_yaml dm_large_encoder --job_idx $JOB_IDX --n_jobs 8 --iqn --shuffle &
#    done
#    wait
#done
#
#
### Shuffled Gridworld, Large Q
#
## Non-IQN models
#for LATENT_SIZE in {8..8}
#do
#    for JOB_IDX in {0..7}
#    do
#        python 01_run_gridworld.py $LATENT_SIZE full_gridsearch --nn_yaml dm_large_q --job_idx $JOB_IDX --n_jobs 8 --discount_factor 0.7 --shuffle &
#    done
#    wait
#done

#for LATENT_SIZE in {8..8}
#do
#    for JOB_IDX in {0..7}
#    do
#        python 01_run_gridworld.py $LATENT_SIZE full_gridsearch --nn_yaml dm_large_q --job_idx $JOB_IDX --n_jobs 8 --shuffle &
#    done
#    wait
#done

# IQN models
for LATENT_SIZE in {8..8}
do
    for JOB_IDX in {0..7}
    do
        python 01_run_gridworld.py $LATENT_SIZE mf_gridsearch --nn_yaml dm_large_q --job_idx $JOB_IDX --n_jobs 8 --discount_factor 0.7 --iqn --shuffle &
    done
    wait
done

#for LATENT_SIZE in {8..8}
#do
#    for JOB_IDX in {0..7}
#    do
#        python 01_run_gridworld.py $LATENT_SIZE mf_gridsearch --nn_yaml dm_large_q --job_idx $JOB_IDX --n_jobs 8 --iqn --shuffle &
#    done
#    wait
#done
