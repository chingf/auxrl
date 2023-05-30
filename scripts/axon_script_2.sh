#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

for JOB_IDX in {0..5}
do
    python run_altT.py $JOB_IDX 6 dm 16 &
done

wait
