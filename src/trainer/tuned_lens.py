import os
from typing import Any, cast

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import Trainer
from transformers.modeling_utils import PreTrainedModel


class TunedLensTrainer(Trainer):
    """Train intermediate-layer translators against the final-layer distribution."""

    def __init__(self, *args, time_epsilon: float = 1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0.0 < time_epsilon < 1.0:
            raise ValueError("time_epsilon must be in (0, 1).")
        self.time_epsilon = float(time_epsilon)

    def create_scheduler(self, num_training_steps, optimizer=None):
        if self.args.lr_scheduler_type == "warmup_stable_decay":
            decay_ratio = getattr(self.args, "decay_ratio", None)
            if decay_ratio is None:
                raise ValueError("decay_ratio is required for warmup_stable_decay.")
            self.args.lr_scheduler_kwargs["num_decay_steps"] = int(
                num_training_steps * decay_ratio
            )
        return super().create_scheduler(num_training_steps, optimizer)

    @torch.no_grad()
    def prediction_step(
        self, model, inputs, prediction_loss_only, ignore_keys=None
    ):
        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return loss.detach(), None, None

        logits = getattr(outputs, "logits", outputs)
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().contiguous()
        labels = inputs.get("labels")
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().contiguous()
        return loss.detach(), logits, labels

    def compute_loss(
        self,
        model,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        del num_items_in_batch
        input_ids = cast(torch.Tensor, inputs["input_ids"])
        labels = cast(torch.Tensor, inputs["labels"])
        attention_mask = cast(torch.Tensor | None, inputs.get("attention_mask"))
        batch_size, sequence_length = input_ids.shape

        timestep = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(
            batch_size, device=input_ids.device
        )
        mask_probability = timestep[:, None].expand(batch_size, sequence_length)
        masked_positions = (
            torch.rand_like(mask_probability) < mask_probability
        ) & labels.ne(-100)
        noised_ids = torch.where(
            masked_positions,
            self.processing_class.mask_token_id,
            input_ids,
        )

        base_model = cast(PreTrainedModel, model.base_model)
        with torch.no_grad():
            outputs = base_model(
                input_ids=noised_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        teacher_probs = outputs.logits.softmax(-1)
        losses = []
        for layer_idx, hidden_state in enumerate(outputs.hidden_states[:-1]):
            student_logits = model(hidden_state, idx=layer_idx)
            token_loss = F.kl_div(
                student_logits.log_softmax(-1),
                teacher_probs,
                reduction="none",
            ).sum(dim=-1)
            if attention_mask is None:
                losses.append(token_loss.mean())
            else:
                losses.append(
                    (token_loss * attention_mask).sum() / attention_mask.sum()
                )
        loss = torch.stack(losses).mean()
        return (loss, outputs) if return_outputs else loss

    def save_model(self, output_dir=None, _internal_call=False):
        del _internal_call
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        model = self.model.module if hasattr(self.model, "module") else self.model
        full_state = model.state_dict()
        state = {
            name: full_state[name].contiguous()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and name in full_state
        }
        save_file(state, os.path.join(output_dir, "model.safetensors"))
