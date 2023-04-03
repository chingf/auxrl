#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

python run_figure8.py 0 16 dm 10 0 &
python run_figure8.py 1 16 dm 10 1 &
python run_figure8.py 2 16 dm 10 2 &
python run_figure8.py 3 16 dm 10 3 &

python run_figure8.py 4 16 dm 10 4 &
python run_figure8.py 5 16 dm 10 5 &
python run_figure8.py 6 16 dm 10 6 &
python run_figure8.py 7 16 dm 10 7 &

python run_figure8.py 8 16 dm 10 0 &
python run_figure8.py 9 16 dm 10 1 &
python run_figure8.py 10 16 dm 10 2 &
python run_figure8.py 11 16 dm 10 3 &
                            
python run_figure8.py 12 16 dm 10 4 &
python run_figure8.py 13 16 dm 10 5 &
python run_figure8.py 14 16 dm 10 6 &
python run_figure8.py 15 16 dm 10 7 &

wait
