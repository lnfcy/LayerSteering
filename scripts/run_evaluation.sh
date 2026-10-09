#!/usr/bin/env bash

# Evaluate LayerSteering on LLaDA Instruct and GSM8K
CUDA_VISIBLE_DEVICES=0 \
HF_ALLOW_CODE_EVAL=1 \
accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    eval.py \
    model=llada-inst \
    dataset.name=gsm8k \
    dataset.n_shot=0 \
    batch_size=1 \
    seed=1234 \
    generation=steering \
    generation.decode_threshold_regular=0.9 \
    generation.tuned_lens_path=checkpoints/llada-inst/tuned_lens/checkpoint-final/model.safetensors \
    'generation.single_token_hidden_bank_paths={20:"artifacts/llada-inst/hidden_bank_l20.pt"}' \
    generation.target_conf_threshold=0.8 \
    generation.prev_target_prob_threshold=0.8 \
    generation.start_block=0 \
    generation.steering_token_filter_mode=Basic
