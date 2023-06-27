#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

for DIM in {2..10}
do
    python 03_gridworld_latents.py $DIM new_gridworld8x8 dm 350
done

wait
