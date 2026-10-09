#!/usr/bin/env bash

# Train the d-Tuned Lens for LLaDA Instruct
CUDA_VISIBLE_DEVICES=0 \
accelerate launch \
    --num_processes 1 \
    train_lens.py \
    model=llada-inst \
    hydra.run.dir=checkpoints/llada-inst/tuned_lens
