#!/bin/bash 

trap "kill 0" EXIT

./bash_script_transfer.sh

wait
