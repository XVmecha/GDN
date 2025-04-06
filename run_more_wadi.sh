#!/bin/bash

# Number of iterations
NUM_ITERATIONS=5  # Change this to your desired number of iterations

for i in $(seq 1 $NUM_ITERATIONS); do
    # Write the training run information to the results file
    echo "training run $i:" >> ./results/results_stats_wadi.txt

    # Run the Python command
    python main.py -dataset wadi -slide_win 5 -slide_stride 1 -batch 32 -epoch 1 -dim 64 -out_layer_num 1 -out_layer_inter_dim 64 -topk 15 -decay 0 -val_ratio 0.2 -device cuda -random_seed 5
    chmod +x ./results/wadi_learned_graph.txt
    mv ./results/wadi_learned_graph.txt ./results/wadi_learned_graph${i}.txt
    done

echo "All training runs completed."