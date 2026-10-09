import copy
import torch
import torch.nn as nn

from typing import cast
from transformers.modeling_utils import PreTrainedModel

def get_final_norm(model: PreTrainedModel) -> nn.Module:
    """Return the final layer normalization module of a LLaDA model."""
    transformer = getattr(getattr(model, "model", None), "transformer", None)
    final_norm = getattr(transformer, "ln_f", None)
    if final_norm is None:
        raise ValueError(
            f"Unsupported model type {type(model)}: expected model.transformer.ln_f."
        )
    return cast(nn.Module, final_norm)


class Unembed(torch.nn.Module):
    """Module that maps transformer hidden states to logits (and vice versa)."""

    final_norm: nn.Module
    unembedding: torch.nn.Linear

    def __init__(self, model: PreTrainedModel):
        """Initialize unembed.

        Args:
            model: A HuggingFace model from which to extract the unembedding matrix.
        """
        super().__init__()

        final_norm = get_final_norm(model)

        if not hasattr(model, "get_output_embeddings"):
            raise ValueError(f"Model {type(model)} does not support get_output_embeddings().")

        unembedding_matrix = model.get_output_embeddings()

        if not isinstance(unembedding_matrix, torch.nn.Linear):
            raise ValueError(
                f"Expected linear output embeddings, got {type(unembedding_matrix)}."
            )

        self.final_norm = copy.deepcopy(final_norm)
        self.unembedding = copy.deepcopy(unembedding_matrix)

        # In general we don't want to finetune the unembed operation.
        self.requires_grad_(False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Convert hidden states into logits."""
        return self.unembedding(self.final_norm(h))
