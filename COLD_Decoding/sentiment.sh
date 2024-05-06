#!/bin/bash

## sentiment analysis

python3 cold_decoding.py \
	--seed 12 \
	--mode sentiment \
	--pretrained_model gpt2-xl \
	--init-temp 1 \
    --length 5 \
	--max-length 50 \
	--num-iters 2000 \
	--min-iters 10 \
	--constraint-weight 0.8 \
	--stepsize 0.1 \
	--noise-iters 1 \
	--win-anneal-iters 1000 \
	--start 0 \
	--end 5 \
	--lr-nll-portion 0.6 \
    --topk 5 \
	--beam-search 0 \
    --output-lgt-temp 1 \
	--verbose \
    --straight-through  \
	--large-noise-iters 50,200,500 \
	--large_gs_std 0.5,0.1,0.05  \
	--input-file "./data/sentiment/test_small.json" \
	--output-dir "./data/sentiment/outputs/" \
	--stepsize-ratio 1  \
    --batch-size 32 \
    --print-every 200 \
	--wandb \
    --sentiment-c2-weight 0.5 \
    --sentiment-c1-weight 0.1 \
    --sentiment-c1-rev-weight 0.1 \
    --sentiment-max-ngram 3