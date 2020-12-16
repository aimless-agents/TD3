#!/bin/bash

# Script to reproduce results

for ((i=0;i<1;i+=1))
do 
	python3.6 tune_td3.py \
	--policy "TD3" \
	--env "HalfCheetahMuJoCoEnv-v0" \
    --smoke_test \
	--seed $i

    # python3.6 tune_td3.py \
	# --policy "TD3" \
	# --env "HalfCheetah-v3" \
    # --save_model \
	# --seed $i
done