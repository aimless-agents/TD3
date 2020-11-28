#!/bin/bash

# Script to reproduce results

for ((i=0;i<1;i+=1))
do 
	python3.6 main.py \
	--policy "TD3" \
	--env "Humanoid-v2" \
    --save_model \
	--seed $i \
	--alpha 0.5
    # --prioritized_replay False
    --start_timesteps 300
done