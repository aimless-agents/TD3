#!/bin/bash

# Script to reproduce results

for ((i=0;i<1;i+=1))
do 
	python3.6 main.py \
	--policy "TD3" \
	--env "Hopper-v3" \
    --discount 0.999 \
    --tau 0.024 \
    --save_model \
	--seed $i
done