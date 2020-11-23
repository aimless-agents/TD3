#!/bin/bash

# Script to reproduce results

for ((i=0;i<1;i+=1))
do 
	python3.6 main.py \
	--policy "TD3" \
	--env "Hopper-v3" \
    --save_model \
	--seed $i \
    # --start_timesteps 300
done