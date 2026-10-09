import argparse
import os
from pathlib import Path

import torch
import transformers
from dotenv import load_dotenv
from omegaconf import OmegaConf
from tqdm import tqdm

from src.utils.models import load_pretrained_model


ROOT = Path(__file__).resolve().parent
MODEL_PATH_ENV = {
    "llada-inst": "LLADA_INST_PATH",
    "llada-1.5": "LLADA_1_5_PATH",
}


def load_model_and_tokenizer(model_name: str):
    load_dotenv(ROOT / ".env")
    env_name = MODEL_PATH_ENV[model_name]
    model_path = os.environ.get(env_name)
    if not model_path:
        raise ValueError(f"Set {env_name} to the local model path.")

    cfg = OmegaConf.create(
        {
            "model": {"name": model_name, "path": model_path},
            "attn_implementation": "eager",
            "generation": {},
        }
    )
    model = load_pretrained_model(
        cfg,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval().cuda()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    tokenizer.add_special_tokens({"mask_token": "<|mdm_mask|>"})
    return model, tokenizer


@torch.inference_mode()
def build_hidden_bank(model, tokenizer, layer_idx: int, batch_size: int):
    vocab_size = int(model.config.vocab_size)
    hidden_size = int(
        getattr(model.config, "d_model", getattr(model.config, "hidden_size"))
    )
    num_layers = int(
        getattr(model.config, "n_layers", getattr(model.config, "num_hidden_layers"))
    )
    if not 0 <= layer_idx < num_layers:
        raise ValueError(f"Layer must be between 0 and {num_layers - 1}.")

    bank = torch.empty(
        vocab_size,
        hidden_size,
        dtype=torch.bfloat16,
        device="cpu",
    )
    mask_token_id = int(tokenizer.mask_token_id)
    device = model.device

    for start in tqdm(range(0, vocab_size, batch_size), desc="Hidden bank"):
        token_ids = torch.arange(
            start,
            min(start + batch_size, vocab_size),
            device=device,
        )
        mask_ids = torch.full_like(token_ids, mask_token_id)
        input_ids = torch.stack((mask_ids, token_ids, mask_ids), dim=1)

        hidden_states = model(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            output_hidden_states=True,
        ).hidden_states
        bank[start : start + token_ids.numel()] = (
            hidden_states[layer_idx + 1][:, 1].to("cpu", dtype=torch.bfloat16)
        )

    return bank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        choices=MODEL_PATH_ENV,
        default="llada-inst",
    )
    parser.add_argument("--layer-idx", type=int, required=True)
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_name)
    bank = build_hidden_bank(
        model,
        tokenizer,
        args.layer_idx,
        args.batch_size,
    )

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"hidden_bank": bank}, out_path)
    print(f"Saved {tuple(bank.shape)} hidden bank to {out_path}")


if __name__ == "__main__":
    main()
