#!/bin/bash 

trap "kill 0" EXIT

./bash_script_transfer.sh
./bash_script_transfer_2.sh

wait
