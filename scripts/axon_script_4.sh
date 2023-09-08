#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

for i in {1..6}; do
    value=$((i * 100 - 50))
    python 03_gridworld_latents.py 10 new_gridworld8x8_shuffobs dm $value
done
