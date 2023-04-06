#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

for JOB_IDX in {0..15}
do
    python run_figure8.py $JOB_IDX 16 dm 10 &
done

wait
