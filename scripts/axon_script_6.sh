#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

python IQN.py 0 1 iqn 8
python IQN.py 0 1 iqn 6

