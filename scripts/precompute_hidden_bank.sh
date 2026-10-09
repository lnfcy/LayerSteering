#!/usr/bin/env bash

# Build the source hidden state bank for layer 20 of LLaDA Instruct
CUDA_VISIBLE_DEVICES=0 \
python precompute_hidden_bank.py \
    --model-name llada-inst \
    --layer-idx 20 \
    --out-path artifacts/llada-inst/hidden_bank_l20.pt \
    --batch-size 512
