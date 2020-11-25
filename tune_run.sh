#!/bin/bash

# Script to reproduce results

for ((i=0;i<1;i+=1))
do 
	python3.6 tune_td3.py \
	--policy "TD3" \
	--env "Hopper-v3" \
	--seed $i
done