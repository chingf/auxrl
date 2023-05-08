#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

for JOB_IDX in {0..1}
do
    python run_gridworld.py $JOB_IDX 2 dm 10 &
done

wait
