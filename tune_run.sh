# range syntax: (seq <start> <step> <end, exclusive>)
# sanity check for the range:
# for i in $(seq 2e-2 5e-3 3.1e-2)
# do 
# 	echo $i
# done

for i in $(seq 2e-2 5e-3 3.1e-2)
do 
	python3.6 main.py \
	--policy "TD3" \
	--custom_env \
	--use_hindsight \
	--run_type "local" \
	--save_model \
	--reacher_epsilon $i \
	--seed 0
done
