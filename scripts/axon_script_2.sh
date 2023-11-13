#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

for JOB_IDX in {0..7}
do
    python 04_altT_pomdp.py $JOB_IDX 8 dm 32 &
done
wait

