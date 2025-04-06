#!/bin/bash

# Number of iterations
NUM_ITERATIONS=10  # Change this to your desired number of iterations

for i in $(seq 1 $NUM_ITERATIONS); do
    # Write the training run information to the results file
    echo "training run $i:" >> ./results/swat/results_stats_swat.txt

    # Run the Python command
    python main.py -dataset swat -slide_win 5 -slide_stride 1 -batch 32 -epoch 50 -dim 64 -out_layer_num 1 -out_layer_inter_dim 64 -topk 15 -decay 0 -val_ratio 0.2 -device cuda
    chmod +x ./results/swat/swat_learned_graph.txt
    mv ./results/swat/swat_learned_graph.txt ./results/swat/swat_learned_graph${i}.txt
    chmod +x ./results/swat/fn_timesteps.txt
    mv ./results/swat/fn_timesteps.txt ./results/swat/fn_timesteps${i}.txt
    chmod +x ./results/swat/fp_timesteps.txt
    mv ./results/swat/fp_timesteps.txt ./results/swat/fp_timesteps${i}.txt
    chmod +x ./results/swat/attack_info.json
    mv ./results/swat/attack_info.json ./results/swat/attack_info${i}.json

    done
    # Number of iterations
NUM_ITERATIONS=10  # Change this to your desired number of iterations

for i in $(seq 1 $NUM_ITERATIONS); do
    # Write the training run information to the results file
    echo "training run $i:" >> ./results/wadi/results_stats_wadi.txt

    # Run the Python command
    python main.py -dataset wadi -slide_win 5 -slide_stride 1 -batch 32 -epoch 50 -dim 128 -out_layer_num 1 -out_layer_inter_dim 128 -topk 30 -decay 0 -val_ratio 0.2 -device cuda
    chmod +x ./results/wadi/wadi_learned_graph.txt
    mv ./results/wadi/wadi_learned_graph.txt ./results/wadi/wadi_learned_graph${i}.txt
    chmod +x ./results/wadi/fn_timesteps.txt
    mv ./results/wadi/fn_timesteps.txt ./results/wadi/fn_timesteps${i}.txt
    chmod +x ./results/wadi/fp_timesteps.txt
    mv ./results/wadi/fp_timesteps.txt ./results/wadi/fp_timesteps${i}.txt
    chmod +x ./results/wadi/attack_info.json
    mv ./results/wadi/attack_info.json ./results/wadi/attack_info${i}.json
    done

echo "All training runs completed."