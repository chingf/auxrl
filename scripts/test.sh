#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT


# Non-IQN models
for LATENT_SIZE in {5..5}
do
    echo $LATENT_SIZE
done
