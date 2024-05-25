#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

for JOB_IDX in {0..7}
do
    python 08_run_poorttask.py $JOB_IDX 8 dm 64 0.01 &
done
wait

for JOB_IDX in {0..7}
do
    python 08_run_poorttask.py $JOB_IDX 8 dm 16 0.01 &
done
wait

for JOB_IDX in {0..7}
do
    python 08_run_poorttask.py $JOB_IDX 8 dm 24 0.01 &
done
wait

for JOB_IDX in {0..7}
do
    python 08_run_poorttask.py $JOB_IDX 8 dm 64 0.05 &
done
wait

for JOB_IDX in {0..7}
do
    python 08_run_poorttask.py $JOB_IDX 8 dm 16 0.05 &
done
wait
