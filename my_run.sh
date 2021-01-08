#!/bin/bash

# Script to reproduce results

for ((i=0;i<1;i+=1))
do 
	python3.6 main.py \
	--policy "TD3" \
	--custom_env \
	--seed $i \
	--prioritized_replay
done