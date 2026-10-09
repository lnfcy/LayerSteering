import os
from collections.abc import Mapping

import torch
import torch.nn.functional as F
import torch.nn as nn
import transformers
from safetensors import safe_open
from transformers.modeling_utils import PreTrainedModel

from src.forward_context import ForwardPassState
from src.frame import INVALID_TOKEN_ID, Frame, FrameDelta, DecodeRecord
from src.generation.utils import (
    check_can_generate,
    greedy_tokens,
)
from src.lens import TunedLens
from src.lens.unembed import Unembed, get_final_norm
from src.models import LLaDAModelLM

_V1_STOPWORDS = {"the", "in", "of", "to", "that"}
_EXTENDED_STOPWORDS = _V1_STOPWORDS | {
    "a",
    "an",
    "and",
    "is",
    "are",
    "for",
    "with",
    "on",
    "as",
    "by",
    "from",
    "at",
    "be",
}
_HUMANEVAL_GUARD_WORDS = {
    "assert",
    "test",
    "tests",
    "case",
    "cases",
    "python",
    "function",
    "verify",
    "verification",
    "should",
}
_MATH_GUARD_WORDS = _EXTENDED_STOPWORDS | {
    "number",
    "total",
    "step",
    "steps",
    "first",
    "next",
    "therefore",
    "thus",
}
_MATH_GUARD_TOKENS = {".", ",", ":"}


def _is_filtered_steering_token(token_text: str, *, mode: str = "Basic") -> bool:
    # Filtering is intentionally lightweight and configurable because code tasks
    # can benefit from structural tokens (newline/indent), while text reasoning
    # often benefits from suppressing low-information function words.
    normalized = token_text.strip().lower()
    if ("�" in token_text) or ("<|" in token_text) or ("|>" in token_text):
        return True

    mode = str(mode)
    if mode == "None":
        return False
    if mode == "Basic":
        return (token_text == "\n") or (token_text.strip() == "") or (normalized in _V1_STOPWORDS)
    if mode == "Code":
        return (
            (token_text == "\n")
            or (token_text.strip() == "")
            or (normalized in _V1_STOPWORDS)
            or (normalized in _HUMANEVAL_GUARD_WORDS)
            or (token_text in {"```"})
        )
    if mode == "Math":
        return (
            (token_text == "\n")
            or (token_text.strip() == "")
            or (normalized in _MATH_GUARD_WORDS)
            or (token_text in _MATH_GUARD_TOKENS)
        )

    raise ValueError(
        f"Unknown steering_token_filter_mode={mode!r}. "
        "Expected one of {'Basic', 'None', 'Code', 'Math'}."
    )


class SteeringState:
    def __init__(
        self,
        model: LLaDAModelLM,
        tuned_lens: TunedLens,
        hidden_banks: dict[int, torch.Tensor],
        prompt_length: int,
        response_mask: torch.Tensor,
        block_mask: torch.Tensor,
        prev_step_final_top1_ids: torch.Tensor | None,
        prev_step_final_top1_probs: torch.Tensor | None,
        mask_token_id: int,
        target_conf_threshold: float,
        commit_conf_threshold: float,
        prev_target_prob_threshold: float,
        invalid_token_ids: torch.Tensor,
        filtered_token_ids: torch.Tensor,
    ):
        self.model = model
        self.tuned_lens = tuned_lens
        self.hidden_banks = hidden_banks
        self.prompt_length = prompt_length
        self.response_mask = response_mask
        self.block_mask = block_mask
        self.prev_step_final_top1_ids = prev_step_final_top1_ids
        self.prev_step_final_top1_probs = prev_step_final_top1_probs
        self.mask_token_id = int(mask_token_id)
        self.target_conf_threshold = float(target_conf_threshold)
        self.commit_conf_threshold = float(commit_conf_threshold)
        self.prev_target_prob_threshold = float(prev_target_prob_threshold)
        self.injected_mask = torch.zeros_like(response_mask, dtype=torch.bool)
        self.injected_target_ids = torch.full_like(
            response_mask, fill_value=self.mask_token_id, dtype=torch.long
        )
        self.invalid_token_ids = invalid_token_ids
        self.filtered_token_ids = filtered_token_ids

    def on_block_output(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        if int(layer_idx) not in self.hidden_banks:
            return x

        candidate_mask = self.response_mask & self.block_mask & ~self.injected_mask
        if self.prev_target_prob_threshold > 0.0:
            if (
                self.prev_step_final_top1_ids is None
                or self.prev_step_final_top1_probs is None
            ):
                return x
            candidate_mask &= (
                self.prev_step_final_top1_probs >= self.prev_target_prob_threshold
            )
            candidate_mask &= ~self.filtered_token_ids[
                self.prev_step_final_top1_ids
            ]
        if not torch.any(candidate_mask):
            return x

        x_gen = x[:, self.prompt_length :, :]
        changed = False

        for batch_idx in range(x_gen.size(0)):
            position_idx = torch.nonzero(
                candidate_mask[batch_idx], as_tuple=False
            ).squeeze(-1)
            if position_idx.numel() == 0:
                continue

            packed_hidden = x_gen[batch_idx : batch_idx + 1, position_idx, :]
            packed_hidden = packed_hidden.to(dtype=self.model.dtype)
            intermediate_logits = self.tuned_lens.forward(packed_hidden, layer_idx)
            logits = intermediate_logits.to(torch.float32)
            if self.invalid_token_ids.numel() > 0:
                logits[..., self.invalid_token_ids] = torch.finfo(logits.dtype).min

            probs = torch.softmax(logits, dim=-1)[0]
            top1_probs, top1_ids = torch.max(probs, dim=-1)

            candidate_local_mask = top1_probs >= self.target_conf_threshold
            prev_target_probs = torch.zeros_like(top1_probs)
            if (
                self.prev_step_final_top1_ids is not None
                and self.prev_step_final_top1_probs is not None
            ):
                prev_ids = self.prev_step_final_top1_ids[batch_idx, position_idx]
                prev_probs = self.prev_step_final_top1_probs[batch_idx, position_idx]
                prev_target_probs = torch.where(
                    prev_ids == top1_ids,
                    prev_probs.to(dtype=top1_probs.dtype),
                    prev_target_probs,
                )

            if self.prev_target_prob_threshold > 0.0:
                candidate_local_mask &= (
                    prev_target_probs >= self.prev_target_prob_threshold
                )
            candidate_local_mask &= ~self.filtered_token_ids[top1_ids]

            candidate_local_indices = torch.nonzero(
                candidate_local_mask,
                as_tuple=False,
            ).squeeze(-1)
            if candidate_local_indices.numel() == 0:
                continue

            combined_scores = (
                0.8 * prev_target_probs.to(torch.float64)
                + 0.2 * top1_probs.to(torch.float64)
            )
            tied_indices = candidate_local_indices
            for values in (
                combined_scores,
                prev_target_probs.to(torch.float64),
                top1_probs.to(torch.float64),
            ):
                tied_values = values[tied_indices]
                tied_indices = tied_indices[tied_values == torch.max(tied_values)]
            best_local_idx = tied_indices[0]
            pos_gen = position_idx[best_local_idx]
            tid = top1_ids[best_local_idx]

            if not changed:
                x = x.clone()
                x_gen = x[:, self.prompt_length :, :]
                changed = True
            x_gen[batch_idx, pos_gen] = self.hidden_banks[int(layer_idx)][tid].to(
                dtype=x_gen.dtype
            )
            self.injected_mask[batch_idx, pos_gen] = True
            self.injected_target_ids[batch_idx, pos_gen] = tid

        if changed:
            x[:, self.prompt_length :, :] = x_gen
        return x

    def build_commit_mask_and_ids(self, probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        commit_mask = self.injected_mask.clone()
        if not torch.any(commit_mask):
            return commit_mask, self.injected_target_ids

        if self.commit_conf_threshold <= 0.0:
            return commit_mask, self.injected_target_ids

        safe_target_ids = self.injected_target_ids.masked_fill(~commit_mask, 0)
        target_probs = torch.gather(
            probs, dim=-1, index=safe_target_ids.unsqueeze(-1)
        ).squeeze(-1)
        commit_mask &= target_probs >= self.commit_conf_threshold

        return commit_mask, self.injected_target_ids


def _get_steering_tokenizer(model: PreTrainedModel):
    tokenizer = getattr(model, "_steering_tokenizer", None)
    if tokenizer is not None:
        return tokenizer

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model.name_or_path,
        trust_remote_code=True,
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"mask_token": "<|mdm_mask|>"})
    model._steering_tokenizer = tokenizer
    return tokenizer


def _load_tuned_lens_translators(
    tuned_lens: TunedLens,
    model: PreTrainedModel,
    lens_path: str,
    layer_ids: set[int],
) -> None:
    with safe_open(lens_path, framework="pt", device="cpu") as checkpoint:
        keys = set(checkpoint.keys())
        for layer_idx in layer_ids:
            prefix = f"layer_translators.{layer_idx}"
            weight_key = f"{prefix}.weight"
            bias_key = f"{prefix}.bias"
            if weight_key not in keys:
                raise ValueError(
                    f"Missing translator weights for layer {layer_idx} in {lens_path}."
                )
            state_dict = {"weight": checkpoint.get_tensor(weight_key)}
            if bias_key in keys:
                state_dict["bias"] = checkpoint.get_tensor(bias_key)
            with torch.device("meta"):
                translator = nn.Linear(
                    int(model.config.hidden_size),
                    int(model.config.hidden_size),
                    bias=bias_key in keys,
                    dtype=model.dtype,
                )
            translator.load_state_dict(state_dict, assign=True)
            tuned_lens.layer_translators[layer_idx] = translator.to(
                device=model.device,
                dtype=model.dtype,
            )


def _get_tuned_lens(
    model: PreTrainedModel,
    lens_path: str,
    active_layer_ids: set[int],
):
    tuned_lens = getattr(model, "_steering_tuned_lens", None)
    tuned_lens_path = getattr(model, "_steering_tuned_lens_path", None)
    if tuned_lens is not None and tuned_lens_path == lens_path:
        loaded_layers = getattr(model, "_steering_tuned_lens_layers", set())
        missing_layers = active_layer_ids - loaded_layers
        _load_tuned_lens_translators(
            tuned_lens, model, lens_path, missing_layers
        )
        model._steering_tuned_lens_layers = loaded_layers | active_layer_ids
        return tuned_lens

    tuned_lens = TunedLens.__new__(TunedLens)
    nn.Module.__init__(tuned_lens)
    object.__setattr__(tuned_lens, "base_model", model)

    unembed = Unembed.__new__(Unembed)
    nn.Module.__init__(unembed)
    unembed.final_norm = get_final_norm(model)
    unembed.unembedding = model.get_output_embeddings()
    tuned_lens.unembed = unembed
    tuned_lens.layer_translators = nn.ModuleList(
        nn.Identity() for _ in range(int(model.config.num_hidden_layers))
    )
    _load_tuned_lens_translators(
        tuned_lens, model, lens_path, active_layer_ids
    )
    tuned_lens.eval()
    object.__setattr__(model, "_steering_tuned_lens", tuned_lens)
    model._steering_tuned_lens_path = lens_path
    model._steering_tuned_lens_layers = set(active_layer_ids)
    return tuned_lens


def _get_hidden_banks(
    model: PreTrainedModel,
    raw_paths: Mapping,
) -> dict[int, torch.Tensor]:
    paths = {int(layer): str(path) for layer, path in raw_paths.items()}
    if getattr(model, "_steering_bank_paths", None) == paths:
        return model._steering_hidden_banks

    banks = {}
    for layer, path in paths.items():
        payload = torch.load(path, map_location="cpu")
        banks[layer] = payload["hidden_bank"].to(
            device=model.device,
            dtype=model.dtype,
        )
    model._steering_bank_paths = paths
    model._steering_hidden_banks = banks
    return banks


def _get_steering_filter_mask(
    model: PreTrainedModel,
    mode: str,
) -> torch.Tensor:
    masks = getattr(model, "_steering_filter_masks", None)
    if masks is None:
        masks = {}
        model._steering_filter_masks = masks
    if mode not in masks:
        tokenizer = _get_steering_tokenizer(model)
        masks[mode] = torch.tensor(
            [
                _is_filtered_steering_token(
                    tokenizer.decode([token_id]),
                    mode=mode,
                )
                for token_id in range(int(model.config.vocab_size))
            ],
            dtype=torch.bool,
            device=model.device,
        )
    return masks[mode]


@torch.no_grad()
def preload_steering_artifacts(
    model: PreTrainedModel,
    tuned_lens_path: str,
    single_token_hidden_bank_paths: Mapping,
    steering_token_filter_mode: str,
) -> None:
    paths = {
        int(layer): str(path)
        for layer, path in single_token_hidden_bank_paths.items()
    }
    _get_steering_tokenizer(model)
    _get_tuned_lens(model, tuned_lens_path, set(paths))
    _get_hidden_banks(model, paths)
    _get_steering_filter_mask(model, steering_token_filter_mode)
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)


def steering_confidence_unmasking(
    scores: torch.Tensor,
    transfer_index_mask: torch.Tensor,
    threshold_regular: float,
) -> tuple[torch.Tensor, ...]:
    confidence = torch.where(transfer_index_mask, scores, -torch.inf)
    if not torch.any(transfer_index_mask):
        return (torch.empty(0, dtype=torch.long, device=scores.device),)
    selected = torch.nonzero(
        transfer_index_mask[0] & (confidence[0] >= threshold_regular),
        as_tuple=False,
    ).squeeze(-1)
    if selected.numel() == 0:
        selected = torch.argmax(confidence[0]).reshape(1)
    return (selected,)


@torch.inference_mode()
def steering_generate_step(
    model,
    frame: Frame,
    block_mask: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    mask_token_id: int = None,  # type: ignore
    *,
    target_conf_threshold: float,
    commit_conf_threshold: float,
    decode_threshold_regular: float,
    hidden_banks: dict[int, torch.Tensor],
    tuned_lens: TunedLens,
    invalid_token_ids: list[int],
    filtered_token_ids: torch.Tensor,
    prev_step_final_top1_ids: torch.Tensor | None = None,
    prev_step_final_top1_probs: torch.Tensor | None = None,
    prev_target_prob_threshold: float = 0.0,
) -> FrameDelta | None:
    frame = frame.as_batch()
    _, prompt_length = frame.prompts.shape

    can_generate = check_can_generate(
        frame,
        eligible_mask=block_mask,
        stop_until_eos=False,
        mask_token_id=mask_token_id,
        eos_token_id=None,
    )
    if not torch.any(can_generate):
        return None

    remaining_mask = frame.generated_tokens == mask_token_id
    transfer_index_mask = remaining_mask.clone()

    x = torch.cat([frame.prompts, frame.generated_tokens], dim=-1)

    steering_state = SteeringState(
        model=model,
        tuned_lens=tuned_lens,
        hidden_banks=hidden_banks,
        prompt_length=prompt_length,
        response_mask=remaining_mask,
        block_mask=block_mask,
        prev_step_final_top1_ids=prev_step_final_top1_ids,
        prev_step_final_top1_probs=prev_step_final_top1_probs,
        mask_token_id=mask_token_id,
        target_conf_threshold=target_conf_threshold,
        commit_conf_threshold=commit_conf_threshold,
        prev_target_prob_threshold=prev_target_prob_threshold,
        invalid_token_ids=invalid_token_ids,
        filtered_token_ids=filtered_token_ids,
    )
    forward_state = ForwardPassState(model.config, steering_state=steering_state)

    outputs = model(
        x,
        attention_mask=attention_mask,
        past_key_values=forward_state,
    )

    logits = outputs.logits[:, prompt_length:]

    confidence, x0, p = greedy_tokens(logits)
    scores = confidence = torch.where(transfer_index_mask, confidence, -torch.inf)
    final_top1_probs, final_top1_ids = torch.max(p, dim=-1)

    commit_mask, commit_ids = steering_state.build_commit_mask_and_ids(p)
    commit_mask &= transfer_index_mask & block_mask

    remaining_mask = transfer_index_mask & block_mask & ~commit_mask
    transfer_index = steering_confidence_unmasking(
        scores=scores,
        transfer_index_mask=remaining_mask,
        threshold_regular=decode_threshold_regular,
    )

    decoded_tokens = torch.where(transfer_index_mask, x0, INVALID_TOKEN_ID)
    if torch.any(commit_mask):
        decoded_tokens = decoded_tokens.clone()
        decoded_tokens[commit_mask] = commit_ids[commit_mask]
        confidence = confidence.clone()
        safe_commit_ids = commit_ids.masked_fill(~commit_mask, 0)
        commit_probs = torch.gather(p, dim=-1, index=safe_commit_ids.unsqueeze(-1)).squeeze(-1)
        confidence[commit_mask] = commit_probs[commit_mask]

    extra = {
        "steering_candidate_count": int(steering_state.injected_mask.sum().item()),
        "steering_replacement_count": int(steering_state.injected_mask.sum().item()),
        "steering_commit_count": int(commit_mask.sum().item()),
        "final_top1_ids": final_top1_ids,
        "final_top1_probs": final_top1_probs,
    }

    commit_index = torch.nonzero(commit_mask[0], as_tuple=False).squeeze(-1)
    if commit_index.numel() > 0:
        transfer_index = (torch.unique(torch.cat([commit_index, transfer_index[0]])),)
    return FrameDelta(
        transfer_index=transfer_index,
        decoded_tokens=decoded_tokens,
        confidence=confidence,
        extra=extra,
    )


def steering_generate(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    block_length: int = 32,
    gen_length: int = 128,
    mask_token_id: int | None = None,
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
    stop_until_eos: bool = False,
    target_conf_threshold: float = 0.9,
    commit_conf_threshold: float = 0.0,
    decode_threshold_regular: float = 0.9,
    prev_target_prob_threshold: float = 0.0,
    single_token_hidden_bank_paths: Mapping | None = None,
    start_block: int = 0,
    steering_token_filter_mode: str = "Basic",
    tuned_lens_path: str | None = None,
) -> DecodeRecord:
    del stop_until_eos
    if mask_token_id is None and os.environ.get("MASK_TOKEN_ID", None) is None:
        raise ValueError(
            "mask_token_id must be provided either as an argument or an environment variable."
        )
    mask_token_id = mask_token_id or int(os.environ.get("MASK_TOKEN_ID"))  # type: ignore
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    tokenizer = _get_steering_tokenizer(model)
    if not tuned_lens_path:
        raise ValueError("tuned_lens_path is required.")
    if not isinstance(single_token_hidden_bank_paths, Mapping):
        raise ValueError("single_token_hidden_bank_paths is required.")
    hidden_banks = _get_hidden_banks(model, single_token_hidden_bank_paths)
    tuned_lens = _get_tuned_lens(model, tuned_lens_path, set(hidden_banks))
    filtered_token_ids = _get_steering_filter_mask(
        model, steering_token_filter_mode
    )
    invalid_token_ids = set(tokenizer.all_special_ids)
    invalid_token_ids.add(mask_token_id)
    if pad_token_id is not None:
        invalid_token_ids.add(pad_token_id)
    if eos_token_id is not None:
        invalid_token_ids.add(eos_token_id)
    if getattr(tokenizer, "bos_token_id", None) is not None:
        invalid_token_ids.add(int(tokenizer.bos_token_id))
    invalid_token_ids = sorted(int(token_id) for token_id in invalid_token_ids)
    invalid_token_ids_tensor = torch.tensor(
        invalid_token_ids,
        device=model.device,
        dtype=torch.long,
    )

    initial_frame = Frame.create_initial_frame(
        input_ids,
        gen_length=gen_length,
        mask_token_id=mask_token_id,
    ).to(device=model.device, dtype=model.dtype)

    if attention_mask is None and pad_token_id is not None:
        attention_mask = (input_ids != pad_token_id).long()

    if attention_mask is not None and attention_mask.shape == input_ids.shape:
        attention_mask = F.pad(attention_mask, (0, gen_length), value=1).to(
            model.device
        )

    frame = initial_frame
    deltas = []
    prev_step_final_top1_ids = None
    prev_step_final_top1_probs = None
    for block_idx in range(num_blocks):
        block_mask = torch.zeros(
            (input_ids.size(0), gen_length),
            dtype=torch.bool,
            device=model.device,
        )
        block_mask[
            :,
            block_idx * block_length : (block_idx + 1) * block_length,
        ] = True

        block_deltas = []
        while True:
            local_target_conf_threshold = (
                2.0 if block_idx < start_block else target_conf_threshold
            )
            delta = steering_generate_step(
                model=model,
                frame=frame,
                block_mask=block_mask,
                attention_mask=attention_mask,
                mask_token_id=mask_token_id,
                target_conf_threshold=local_target_conf_threshold,
                commit_conf_threshold=commit_conf_threshold,
                decode_threshold_regular=decode_threshold_regular,
                hidden_banks=hidden_banks,
                prev_target_prob_threshold=prev_target_prob_threshold,
                tuned_lens=tuned_lens,
                invalid_token_ids=invalid_token_ids_tensor,
                filtered_token_ids=filtered_token_ids,
                prev_step_final_top1_ids=prev_step_final_top1_ids,
                prev_step_final_top1_probs=prev_step_final_top1_probs,
            )
            if delta is None:
                break

            final_top1_ids = delta.extra.pop("final_top1_ids", None)
            final_top1_probs = delta.extra.pop("final_top1_probs", None)
            if final_top1_ids is not None:
                prev_step_final_top1_ids = final_top1_ids
            if final_top1_probs is not None:
                prev_step_final_top1_probs = final_top1_probs

            cpu_delta = delta.to("cpu")
            block_deltas.append(cpu_delta)
            frame = frame.apply_delta(delta)

        deltas.extend(block_deltas)

    return DecodeRecord(
        initial_frame=initial_frame.to("cpu"),
        deltas=deltas,
        block_length=block_length,
    )
