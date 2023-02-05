#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

python run_foraging_transfer.py 0 16 basic_large_encoder 5 0 &
python run_foraging_transfer.py 1 16 basic_large_encoder 5 1 &
python run_foraging_transfer.py 2 16 basic_large_encoder 5 2 &
python run_foraging_transfer.py 3 16 basic_large_encoder 5 3 &
python run_foraging_transfer.py 4 16 basic_large_encoder 5 4 &
python run_foraging_transfer.py 5 16 basic_large_encoder 5 5 &
python run_foraging_transfer.py 6 16 basic_large_encoder 5 6 &
python run_foraging_transfer.py 7 16 basic_large_encoder 5 7 &
python run_foraging_transfer.py 8 16 basic_large_encoder 5 0 &
python run_foraging_transfer.py 9 16 basic_large_encoder 5 1 &
python run_foraging_transfer.py 10 16 basic_large_encoder 5 2 &
python run_foraging_transfer.py 11 16 basic_large_encoder 5 3 &
python run_foraging_transfer.py 12 16 basic_large_encoder 5 4 &
python run_foraging_transfer.py 13 16 basic_large_encoder 5 5 &
python run_foraging_transfer.py 14 16 basic_large_encoder 5 6 &
python run_foraging_transfer.py 15 16 basic_large_encoder 5 7 &

python run_foraging_transfer.py 0 16 basic_large_decoder 5 0 &
python run_foraging_transfer.py 1 16 basic_large_decoder 5 1 &
python run_foraging_transfer.py 2 16 basic_large_decoder 5 2 &
python run_foraging_transfer.py 3 16 basic_large_decoder 5 3 &
python run_foraging_transfer.py 4 16 basic_large_decoder 5 4 &
python run_foraging_transfer.py 5 16 basic_large_decoder 5 5 &
python run_foraging_transfer.py 6 16 basic_large_decoder 5 6 &
python run_foraging_transfer.py 7 16 basic_large_decoder 5 7 &
python run_foraging_transfer.py 8 16 basic_large_decoder 5 0 &
python run_foraging_transfer.py 9 16 basic_large_decoder 5 1 &
python run_foraging_transfer.py 10 16 basic_large_decoder 5 2 &
python run_foraging_transfer.py 11 16 basic_large_decoder 5 3 &
python run_foraging_transfer.py 12 16 basic_large_decoder 5 4 &
python run_foraging_transfer.py 13 16 basic_large_decoder 5 5 &
python run_foraging_transfer.py 14 16 basic_large_decoder 5 6 &
python run_foraging_transfer.py 15 16 basic_large_decoder 5 7 &

wait
