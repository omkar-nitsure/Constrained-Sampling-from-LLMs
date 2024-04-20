#!/bin/bash

## CommonGen


# # Record the start time
# start_time=$(date +%s.%N)

python3 cold_decoding.py \
	--seed 12 \
	--mode lexical_generation \
	--pretrained_model gpt2-xl  \
	--init-temp 1 \
    --length 10 \
	--max-length 40 \
	--num-iters 2000 \
	--min-iters 1000 \
	--constraint-weight 0.5 \
    --abductive-c2-weight 0.1 \
	--stepsize 0.1 \
	--noise-iters 1 \
	--win-anneal-iters 1000 \
	--start 0 \
	--end 2 \
	--lr-nll-portion 0.3 \
    --topk 5 \
    --output-lgt-temp 1 \
	--verbose \
    --straight-through  \
	--wandb  \
	--large-noise-iters 50,500,1000,1500 \
	--large_gs_std 1,0.5,0.1,0.05  \
	--stepsize-ratio 1  \
    --batch-size 32 \
    --repeat-batch 1 \
    --print-every 200 \
    --input-file "./data/commongen/commongen.dev_small.jsonl" \
	--output-dir "./data/commongen/rlnll_changes/" \
# # Record the end time
# end_time=$(date +%s.%N)

# # Calculate the execution time
# execution_time=$(echo "$end_time - $start_time" | bc)

# # Print the execution time
# echo "Execution time: $execution_time seconds"