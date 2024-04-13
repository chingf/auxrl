#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

#for i in {0..12}; do
for i in {10..12}; do
    value=$((i * 50))
    python 03_gridworld_latents.py 10 iqn3 iqn $value
done
