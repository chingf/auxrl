#!/bin/bash 

trap "kill 0" EXIT

python run_foraging_transfer.py 0 3 simplest_e32d64 6 0 &
python run_foraging_transfer.py 1 3 simplest_e32d64 6 1 &
python run_foraging_transfer.py 2 3 simplest_e32d64 6 2 &

python run_foraging_transfer.py 0 3 simplest_e32d64 8 3 &
python run_foraging_transfer.py 1 3 simplest_e32d64 8 0 &
python run_foraging_transfer.py 2 3 simplest_e32d64 8 1 &

python run_foraging_transfer.py 0 3 simplest_e32d64 10 2 &
python run_foraging_transfer.py 1 3 simplest_e32d64 10 3 &
python run_foraging_transfer.py 2 3 simplest_e32d64 10 0 &

python run_foraging_transfer.py 0 3 simplest_e64d64 4 1 &
python run_foraging_transfer.py 1 3 simplest_e64d64 4 2 &
python run_foraging_transfer.py 2 3 simplest_e64d64 4 3 &

python run_foraging_transfer.py 0 3 simplest_e64d64 6 0 &
python run_foraging_transfer.py 1 3 simplest_e64d64 6 1 &
python run_foraging_transfer.py 2 3 simplest_e64d64 6 2 &

python run_foraging_transfer.py 0 3 simplest_e64d64 7 3 &
python run_foraging_transfer.py 1 3 simplest_e64d64 7 1 &
python run_foraging_transfer.py 2 3 simplest_e64d64 7 2 &

wait
