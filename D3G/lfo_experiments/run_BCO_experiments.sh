#!/bin/bash

# Script to reproduce results
declare -a randomness=("0.0" "0.25" "0.5" "0.75" "1.0")

for ((i=0;i<10;i+=1))
do
    for r in "${randomness[@]}"
    do
        python learn_from_observation.py \
        --policy "D3G" \
        --env "InvertedPendulum-v2" \
        --seed $i \
        --randomness $r \
        --expert \

        python learn_from_observation.py \
        --policy "D3G" \
        --env "Reacher-v2" \
        --seed $i \
        --randomness $r \
        --expert
    done

done
