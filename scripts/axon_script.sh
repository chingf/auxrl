#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

# Internal dim 13
for JOB_IDX in {0..5}
do
    python 01_run_gridworld.py $JOB_IDX 6 dm_large_q 13 &
done
wait

for JOB_IDX in {0..5}
do
    python 02_run_gridworld_frozentransfer.py $JOB_IDX 6 dm_large_q 13 &
done
wait

# Internal dim 14
for JOB_IDX in {0..5}
do
    python 01_run_gridworld.py $JOB_IDX 6 dm_large_q 14 &
done
wait

for JOB_IDX in {0..5}
do
    python 02_run_gridworld_frozentransfer.py $JOB_IDX 6 dm_large_q 14 &
done
wait

# Internal dim 15
for JOB_IDX in {0..5}
do
    python 01_run_gridworld.py $JOB_IDX 6 dm_large_q 15 &
done
wait

for JOB_IDX in {0..5}
do
    python 02_run_gridworld_frozentransfer.py $JOB_IDX 6 dm_large_q 15 &
done
wait

# Internal dim 16
for JOB_IDX in {0..5}
do
    python 01_run_gridworld.py $JOB_IDX 6 dm_large_q 16 &
done
wait

for JOB_IDX in {0..5}
do
    python 02_run_gridworld_frozentransfer.py $JOB_IDX 6 dm_large_q 16 &
done
wait

# Internal dim 17
for JOB_IDX in {0..5}
do
    python 01_run_gridworld.py $JOB_IDX 6 dm_large_q 17 &
done
wait

for JOB_IDX in {0..5}
do
    python 02_run_gridworld_frozentransfer.py $JOB_IDX 6 dm_large_q 17 &
done
wait

# Internal dim 18
for JOB_IDX in {0..5}
do
    python 01_run_gridworld.py $JOB_IDX 6 dm_large_q 18 &
done
wait

for JOB_IDX in {0..5}
do
    python 02_run_gridworld_frozentransfer.py $JOB_IDX 6 dm_large_q 18 &
done
wait
