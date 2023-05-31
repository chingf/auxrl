#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

# Internal dim 4
for JOB_IDX in {0..11}
do
    python run_gridworld.py $JOB_IDX 12 dm_large_encoder 4 &
done

wait

for JOB_IDX in {0..11}
do
    python run_gridworld_fulltransfer.py $JOB_IDX 12 dm_large_encoder 4 &
done

wait

# Internal dim 5
for JOB_IDX in {0..11}
do
    python run_gridworld.py $JOB_IDX 12 dm_large_encoder 5 &
done

wait

for JOB_IDX in {0..11}
do
    python run_gridworld_fulltransfer.py $JOB_IDX 12 dm_large_encoder 5 &
done

wait

# Internal dim 6
for JOB_IDX in {0..11}
do
    python run_gridworld.py $JOB_IDX 12 dm_large_encoder 6 &
done

wait

for JOB_IDX in {0..11}
do
    python run_gridworld_fulltransfer.py $JOB_IDX 12 dm_large_encoder 6 &
done

wait

# Internal dim 6
for JOB_IDX in {0..11}
do
    python run_gridworld.py $JOB_IDX 12 dm_large_encoder 7 &
done

wait

for JOB_IDX in {0..11}
do
    python run_gridworld_fulltransfer.py $JOB_IDX 12 dm_large_encoder 7 &
done

wait

# Internal dim 3
for JOB_IDX in {0..11}
do
    python run_gridworld.py $JOB_IDX 12 dm_large_encoder 3 &
done

wait

for JOB_IDX in {0..11}
do
    python run_gridworld_fulltransfer.py $JOB_IDX 12 dm_large_encoder 3 &
done

wait

# Internal dim 8
for JOB_IDX in {0..11}
do
    python run_gridworld.py $JOB_IDX 12 dm_large_encoder 8 &
done

wait

for JOB_IDX in {0..11}
do
    python run_gridworld_fulltransfer.py $JOB_IDX 12 dm_large_encoder 8 &
done

wait

# Internal dim 9
for JOB_IDX in {0..11}
do
    python run_gridworld.py $JOB_IDX 12 dm_large_encoder 9 &
done

wait

for JOB_IDX in {0..11}
do
    python run_gridworld_fulltransfer.py $JOB_IDX 12 dm_large_encoder 9 &
done

wait

