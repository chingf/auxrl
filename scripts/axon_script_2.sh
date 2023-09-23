#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

for JOB_IDX in {0..1}
do
    python 04_run_altT.py $JOB_IDX 2 dm_selq_v3 96 &
done
wait

python 05_altT_latents.py test_altT dm_selq_v3 96 30

