#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

for JOB_IDX in {0..7}
do
    python 09_run_gridworld_cifar.py $JOB_IDX 8 dm 10 1.0 &
done
wait

for JOB_IDX in {0..7}
do
    python 09_run_gridworld_cifar.py $JOB_IDX 8 dm 8 1.0 &
done
wait

