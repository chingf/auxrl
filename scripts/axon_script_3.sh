#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

21for JOB_IDX in {0..7}
do
    python 07_run_lineartrack.py $JOB_IDX 8 dm 32 &
done
wait

for JOB_IDX in {0..7}
do
    python 07_run_lineartrack.py $JOB_IDX 8 dm 24 &
done
wait
