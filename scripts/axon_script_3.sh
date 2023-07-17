#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

for JOB_IDX in {0..7}
do
    python 06_dicarlo_gridworld_swap.py $JOB_IDX 8 dm 16 &
done
wait

