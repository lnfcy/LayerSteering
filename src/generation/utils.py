import re
import torch

from src.frame import Frame


def decode_final_frame(
    tokenizer, final_frame: Frame, stop_words: list[str] | None = None, **kwargs
) -> str | list[str]:
    """Decode generated tokens and trim text at the first stop word."""
    frame = final_frame.as_batch()

    can_generate = check_can_generate(
        frame,
        stop_until_eos=True,  # we only want to check tokens before the first EOS in this case
        mask_token_id=tokenizer.mask_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if can_generate.any():
        raise ValueError(
            "The frame contains mask tokens, indicating that the generation has not completed."
        )

    stop_words = stop_words or []
    skip_special_tokens = kwargs.pop("skip_special_tokens", True)
    if tokenizer.eos_token not in stop_words:
        stop_words.append(tokenizer.eos_token)

    # trim until stop words
    filtered_tokens = frame.generated_tokens.clone()
    filtered_tokens[frame.generated_tokens > len(tokenizer.get_vocab())] = (
        tokenizer.eos_token_id
    )
    texts = tokenizer.batch_decode(filtered_tokens, skip_special_tokens=False, **kwargs)

    texts = [
        (
            text[: match.start()]
            if (match := re.search(r"|".join(re.escape(sw) for sw in stop_words), text))
            else text
        )
        for text in texts
    ]

    # remove special tokens
    texts = tokenizer.batch_decode(
        tokenizer(texts).input_ids, skip_special_tokens=skip_special_tokens, **kwargs
    )

    return texts if final_frame.is_batched else texts[0]


def greedy_tokens(logits: torch.Tensor):
    probs = torch.softmax(logits, dim=-1)
    confidence, token_ids = probs.max(dim=-1)
    return confidence, token_ids, probs


def check_can_generate(
    frame: Frame,
    eligible_mask: torch.Tensor | None = None,
    stop_until_eos: bool = False,
    mask_token_id: int | None = None,
    eos_token_id: int | None = None,
):
    """Return whether each sequence has an eligible masked position."""
    frame = frame.as_batch()
    batch_size = frame.prompts.shape[0]
    gen_length = frame.generated_tokens.size(1)
    device = frame.generated_tokens.device

    if eligible_mask is None:
        eligible_mask = torch.ones_like(frame.generated_tokens)

    if stop_until_eos:
        if eos_token_id is None:
            raise ValueError(
                "eos_token_id must be specified if stop_until_eos is True."
            )
        eos_mask = frame.generated_tokens == eos_token_id
        first_eos_idx = torch.where(
            eos_mask.any(dim=-1, keepdim=True),
            eos_mask.int().argmax(dim=-1, keepdim=True),
            gen_length,
        )
        eligible_mask &= (
            torch.arange(gen_length, device=device).unsqueeze(0).expand(batch_size, -1)
        ) < first_eos_idx

    if mask_token_id is not None:
        eligible_mask &= frame.generated_tokens == mask_token_id

    return eligible_mask.any(dim=-1)
