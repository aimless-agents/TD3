#!/bin/bash

# Script to reproduce results

for ((i=0;i<1;i+=1))
do 
	python3.6 tune_td3.py \
	--policy "TD3" \
	--env "HalfCheetahMuJoCoEnv-v0" \
    --discount 0.996 \
    --tau 0.0005 \
    --prioritized_replay \
    --use_rank \
	--seed $i

    # python3.6 tune_td3.py \
	# --policy "TD3" \
	# --env "HalfCheetah-v3" \
    # --save_model \
	# --seed $i
done