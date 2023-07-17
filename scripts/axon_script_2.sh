#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

for EP in {0..3}
do
    python 03_gridworld_latents.py 16 dicarlo_swap_postbug_gridworld8x8_shuffobs dm $EP
done

wait
