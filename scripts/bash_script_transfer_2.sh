#!/bin/bash 

trap "kill 0" EXIT

python bash_foraging_transfer_script.py 0 3 simplest_e10d64 4 0 &
python bash_foraging_transfer_script.py 1 3 simplest_e10d64 4 0 &
python bash_foraging_transfer_script.py 2 3 simplest_e10d64 4 0 &

python bash_foraging_transfer_script.py 0 3 simplest_e10d64 5 1 &
python bash_foraging_transfer_script.py 1 3 simplest_e10d64 5 1 &
python bash_foraging_transfer_script.py 2 3 simplest_e10d64 5 1 &

python bash_foraging_transfer_script.py 0 3 simplest_e10d64 6 2 &
python bash_foraging_transfer_script.py 1 3 simplest_e10d64 6 2 &
python bash_foraging_transfer_script.py 2 3 simplest_e10d64 6 2 &

python bash_foraging_transfer_script.py 0 3 simplest_e10d64 7 3 &
python bash_foraging_transfer_script.py 1 3 simplest_e10d64 7 3 &
python bash_foraging_transfer_script.py 2 3 simplest_e10d64 7 3 &

wait
