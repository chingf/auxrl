#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

### Frozen Transfer: Shuffled Gridworld, Large Q

## Non-IQN models
#for LATENT_SIZE in {5..24}
#do
#    for JOB_IDX in {0..15}
#    do
#        python 02_run_gridworld_frozentransfer.py $LATENT_SIZE --job_idx $JOB_IDX --n_jobs 16 &
#    done
#    wait
#done

# IQN models
for LATENT_SIZE in {5..24}
do
    for JOB_IDX in {0..15}
    do
        python 02_run_gridworld_frozentransfer.py $LATENT_SIZE --job_idx $JOB_IDX --n_jobs 16 --discount_factor 0.7 --iqn &
    done
    wait
done


# ### Shuffled Gridworld, Large Q
# 
# ## Non-IQN models
# 
# for LATENT_SIZE in {5..24}
# do
#     for JOB_IDX in {0..15}
#     do
#         python 01_run_gridworld.py $LATENT_SIZE selected_models_large_q --nn_yaml dm_large_q --job_idx $JOB_IDX --n_jobs 16 --shuffle &
#     done
#     wait
# done
# 
# # IQN models
# for LATENT_SIZE in {5..24}
# do
#     for JOB_IDX in {0..15}
#     do
#         python 01_run_gridworld.py $LATENT_SIZE mf1 --nn_yaml dm_large_q --job_idx $JOB_IDX --n_jobs 16 --discount_factor 0.7 --iqn --shuffle &
#     done
#     wait
# done
