#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

for JOB_IDX in {0..55}
do
    python run_foraging.py $JOB_IDX 56 dm 10 &
done

wait
