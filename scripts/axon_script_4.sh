#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

for i in {0..6}; do
    value=$((i * 100))
    python 03_gridworld_latents.py 10 iqn_shuffobs iqn $value
done

#for i in {1..6}; do
#    value=$((i * 100 - 50))
#    python 03_gridworld_latents.py 10 iqn_shuffobs iqn $value
#done
