#!/bin/bash

# Script to reproduce results

for ((i=0;i<1;i+=1))
do 
	python3.6 main.py \
	--policy "TD3" \
	--env "ReacherPyBulletEnv-v0" \
	--custom_env \
	--use_hindsight \
	--seed $i

    # python3.6 tune_td3.py \
	# --policy "TD3" \
	# --env "HalfCheetah-v3" \
    # --save_model \
	# --seed $i
done
