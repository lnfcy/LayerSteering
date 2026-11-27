# Final d-Tuned Lens checkpoints

This directory contains the final d-Tuned Lens weights for LLaDA-8B-Instruct
(`llada-inst`) and LLaDA-1.5 (`llada-1.5`). Each original
`model.safetensors` is split into 11 contiguous binary parts. The
[manifest](manifest.json) gives byte counts and SHA-256 checksums for each part
and the original files.

## Restore from a repository copy

From the repository root, run the following with Python 3.11 or later.

```sh
python3 checkpoints/restore.py
```

## Download from the anonymous review repository

Save [restore.py](restore.py) in a local `checkpoints` directory. Replace
`ANONYMOUS_REPOSITORY_URL` with the URL in the paper, then run this from the
parent directory of `checkpoints`.

```sh
python3 checkpoints/restore.py --repo-url 'ANONYMOUS_REPOSITORY_URL'
```

The script verifies each part and the reconstructed file. It writes the
restored weights to the following paths.

- `checkpoints/llada-inst/tuned_lens/checkpoint-final/model.safetensors`
- `checkpoints/llada-1.5/tuned_lens/checkpoint-final/model.safetensors`

To restore just one checkpoint, add `--model llada-inst` or
`--model llada-1.5`. Use the resulting `model.safetensors` path as
`generation.tuned_lens_path` in your evaluation configuration.
