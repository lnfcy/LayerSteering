import torch
import torch.nn as nn

from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class AttentionContext:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    residual: torch.Tensor
    o: torch.Tensor | None = None
    attn_weight: torch.Tensor | None = None
    q_position_ids: torch.Tensor | None = None
    kv_position_ids: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None

    @classmethod
    def convert_attention_mask(
        cls,
        attention_mask: torch.Tensor | None,
        dtype: torch.dtype,
        query_length: int | None = None,
        key_value_length: int | None = None,
    ):
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :].expand(
                    attention_mask.size(0),
                    1,
                    query_length or attention_mask.size(1),
                    key_value_length or attention_mask.size(1),
                )
            elif attention_mask.dim() != 4:
                raise ValueError(
                    f"Expected attention_mask to have 2 or 4 dimensions, but got {attention_mask.dim()}."
                )

            attention_mask = (1.0 - attention_mask.to(dtype)) * torch.finfo(dtype).min

        return attention_mask


@dataclass
class FFNContext:
    x: torch.Tensor
    residual: torch.Tensor
    ffn_out: torch.Tensor | None = None


@dataclass
class ModelForwardContext:
    x: torch.Tensor
    logits: torch.Tensor | None = None


class ForwardPassState:
    """
    Minimal execution context used to keep the model forward path uniform without
    enabling any reusable cache behavior.
    """

    def __init__(self, model_config, steering_state=None):
        self.model_config = model_config
        self.steering_state = steering_state

    @contextmanager
    def model_forward(self, x: torch.Tensor):
        input_shape = x.shape
        ctx = ModelForwardContext(x=x)

        yield ctx

        if ctx.logits is None:
            raise RuntimeError("The logits are not set in the context.")

        if ctx.logits.shape[:2] != input_shape[:2]:
            raise RuntimeError(
                f"The logits shape {ctx.logits.shape!r} is not compatible with the input shape {input_shape!r}."
            )

    @contextmanager
    def attention(
        self,
        layer_idx: int,
        x: torch.Tensor,
        attn_norm: nn.Module,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ):
        residual = x
        x = attn_norm(x)
        if x.numel() > 0:
            q, k, v = q_proj(x), k_proj(x), v_proj(x)
        else:
            q, k, v = x[:, 0:0], x[:, 0:0], x[:, 0:0]

        ctx = AttentionContext(
            q=q,
            k=k,
            v=v,
            residual=residual,
            attention_mask=AttentionContext.convert_attention_mask(
                attention_mask,
                dtype=q.dtype,
                query_length=q.shape[1],
                key_value_length=k.shape[1],
            ),
            q_position_ids=position_ids,
            kv_position_ids=position_ids,
        )
        if self.steering_state is not None and hasattr(
            self.steering_state, "on_attention"
        ):
            ctx = self.steering_state.on_attention(
                layer_idx=layer_idx,
                ctx=ctx,
                attn_norm=attn_norm,
                q_proj=q_proj,
                k_proj=k_proj,
                v_proj=v_proj,
            )
        yield ctx

        if ctx.o is None:
            raise RuntimeError("The attention output is not set in the context.")

        if ctx.residual.shape != ctx.o.shape:
            raise RuntimeError(
                f"The attention output shape {ctx.o.shape!r} is not compatible with the residual shape {ctx.residual.shape!r}."
            )

    @contextmanager
    def ffn(self, layer_idx: int, x: torch.Tensor):
        ctx = FFNContext(x=x, residual=x)
        yield ctx

        if ctx.ffn_out is None:
            raise RuntimeError(
                "The feed-forward network output is not set in the context."
            )

    def on_block_output(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        if self.steering_state is None:
            return x
        return self.steering_state.on_block_output(layer_idx, x)
