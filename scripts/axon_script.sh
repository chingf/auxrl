#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

python run_foraging.py 0 16 basic_large_encoder 5 0 &
python run_foraging.py 1 16 basic_large_encoder 5 1 &
python run_foraging.py 2 16 basic_large_encoder 5 2 &
python run_foraging.py 3 16 basic_large_encoder 5 3 &
python run_foraging.py 4 16 basic_large_encoder 5 4 &
python run_foraging.py 5 16 basic_large_encoder 5 5 &
python run_foraging.py 6 16 basic_large_encoder 5 6 &
python run_foraging.py 7 16 basic_large_encoder 5 7 &
python run_foraging.py 8 16 basic_large_encoder 5 0 &
python run_foraging.py 9 16 basic_large_encoder 5 1 &
python run_foraging.py 10 16 basic_large_encoder 5 2 &
python run_foraging.py 11 16 basic_large_encoder 5 3 &
python run_foraging.py 12 16 basic_large_encoder 5 4 &
python run_foraging.py 13 16 basic_large_encoder 5 5 &
python run_foraging.py 14 16 basic_large_encoder 5 6 &
python run_foraging.py 15 16 basic_large_encoder 5 7 &

python run_foraging.py 0 16 basic_large_decoder 5 0 &
python run_foraging.py 1 16 basic_large_decoder 5 1 &
python run_foraging.py 2 16 basic_large_decoder 5 2 &
python run_foraging.py 3 16 basic_large_decoder 5 3 &
python run_foraging.py 4 16 basic_large_decoder 5 4 &
python run_foraging.py 5 16 basic_large_decoder 5 5 &
python run_foraging.py 6 16 basic_large_decoder 5 6 &
python run_foraging.py 7 16 basic_large_decoder 5 7 &
python run_foraging.py 8 16 basic_large_decoder 5 0 &
python run_foraging.py 9 16 basic_large_decoder 5 1 &
python run_foraging.py 10 16 basic_large_decoder 5 2 &
python run_foraging.py 11 16 basic_large_decoder 5 3 &
python run_foraging.py 12 16 basic_large_decoder 5 4 &
python run_foraging.py 13 16 basic_large_decoder 5 5 &
python run_foraging.py 14 16 basic_large_decoder 5 6 &
python run_foraging.py 15 16 basic_large_decoder 5 7 &

wait
