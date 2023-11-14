#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

for JOB_IDX in {0..7}
do
    python 10_run_gridworld_stochastic.py $JOB_IDX 8 dm 10 &
done
wait

