#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

python run_figure8.py 0 4 dm 16 0 &
python run_figure8.py 1 4 dm 16 1 &
python run_figure8.py 2 4 dm 16 2 &
python run_figure8.py 3 4 dm 16 3 &

wait
