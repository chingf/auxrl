#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

python 05_altT_latents.py test_pomdp dm 64 60

