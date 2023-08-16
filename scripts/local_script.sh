#!/bin/bash 

python 04_lsi_len1.py -1 1 dm_small_encoder_large_q 64
python 04_lsi_len1.py -1 1 dm_small_encoder_large_q 96
python 04_lsi_len1.py -1 1 dm_small_encoder_large_q 32

wait
