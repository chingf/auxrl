#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

python run_foraging.py 0 8 dm 24 0 &
python run_foraging.py 1 8 dm 24 1 &
python run_foraging.py 2 8 dm 24 2 &
python run_foraging.py 3 8 dm 24 3 &

python run_foraging.py 4 8 dm 24 0 &
python run_foraging.py 5 8 dm 24 1 &
python run_foraging.py 6 8 dm 24 2 &
python run_foraging.py 7 8 dm 24 3 &

wait
