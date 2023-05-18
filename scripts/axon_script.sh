#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

for JOB_IDX in {0..4}
do
    python run_gridworld_frozentransfer.py $JOB_IDX 5 dm 10 &
done

wait
