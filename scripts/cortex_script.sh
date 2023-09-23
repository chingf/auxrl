#!/bin/bash 

# Last run was 14-21

for LATENT in {5..21}
do
    python 01_run_gridworld.py -1 1 dm_large_q $LATENT 1.0 shuffle
done

for LATENT in {5..21}
do
    python 02_run_gridworld_frozentransfer.py -1 1 dm_large_q $LATENT shuffle
done
