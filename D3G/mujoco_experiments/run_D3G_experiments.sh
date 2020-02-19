#!/bin/bash

# Script to reproduce results

for ((i=0;i<10;i+=1))
do 
	python main.py \
	--policy "D3G" \
	--env "HalfCheetah-v2" \
	--seed $i \
	--start_timesteps 10000 \

	python main.py \
	--policy "D3G" \
	--env "Hopper-v2" \
	--seed $i \
	--start_timesteps 10000 \

	python main.py \
	--policy "D3G" \
	--env "Walker2d-v2" \
	--seed $i \
	--start_timesteps 10000 \

	python main.py \
	--policy "D3G" \
	--env "Ant-v2" \
	--seed $i \
	--start_timesteps 10000 \

	python main.py \
	--policy "D3G" \
	--env "InvertedPendulum-v2" \
	--seed $i \
	--start_timesteps 10000 \

	python main.py \
	--policy "D3G" \
	--env "InvertedDoublePendulum-v2" \
	--seed $i \
	--start_timesteps 10000 \

	python main.py \
	--policy "D3G" \
	--env "Reacher-v2" \
	--seed $i \
	--start_timesteps 10000 \

	python main.py \
	--policy "D3G" \
	--env "Humanoid-v2" \
	--seed $i \
	--start_timesteps 10000 
done
