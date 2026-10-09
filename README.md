## LayerSteering

This repository provides the anonymous implementation of LayerSteering for
LLaDA Instruct and LLaDA 1.5.

## Setup

```bash
conda create -n layersteering python=3.11 -y
conda activate layersteering
pip install -r requirements/common.txt
cp .env.example .env
```


## Prepare

Train the d-Tuned Lens for LLaDA Instruct:

```bash
bash scripts/train_d_tuned_lens.sh
```

The trained checkpoint is saved to
`checkpoints/llada-inst/tuned_lens/checkpoint-final/model.safetensors`.

Build the layer 20 hidden state bank for LLaDA Instruct:

```bash
bash scripts/precompute_hidden_bank.sh
```

The bank is saved to `artifacts/llada-inst/hidden_bank_l20.pt`. Build one bank
for every steering layer by changing `--layer-idx` and `--out-path` in the
script. If GPU memory is limited, reduce `--batch-size`.

## Evaluation

Run LayerSteering:

```bash
bash scripts/run_evaluation.sh
```

## Supported Configurations

Models:

- `llada-inst`
- `llada-1.5`

Tasks:

- `gsm8k`
- `math-500`
- `humaneval_instruct`
- `mbpp_instruct`
