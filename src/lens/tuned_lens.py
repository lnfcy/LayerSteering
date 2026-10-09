import torch
import torch.nn as nn

from typing import Generator
from transformers.modeling_utils import PreTrainedModel
from safetensors.torch import load_file as torch_load

from .unembed import Unembed


class TunedLens(nn.Module):
    """A tuned lens for decoding hidden states into logits."""

    base_model: PreTrainedModel
    unembed: Unembed
    layer_translators: torch.nn.ModuleList

    def __init__(self, model: PreTrainedModel, bias: bool = True):
        """Create a TunedLens.

        Args:
            unembed: The unembed operation to use.
            config: The configuration for this lens.
        """
        super().__init__()
        # The unembedding might be int8 if we're using bitsandbytes
        config = model.config
        self.base_model = model
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.unembed = Unembed(model)

        # Don't include the final layer since it does not need a translator
        self.layer_translators = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    config.hidden_size, config.hidden_size, dtype=model.dtype, bias=bias
                )
                for _ in range(config.num_hidden_layers)
            ]
        )

        # Identity initialization
        for translator in self.layer_translators:
            nn.init.zeros_(translator.weight)  # type: ignore
            if translator.bias is not None:
                nn.init.zeros_(translator.bias)  # type: ignore

    def __getitem__(self, item: int) -> torch.nn.Module:
        """Get the probe module at the given index."""
        return self.layer_translators[item]

    def __iter__(self) -> Generator[torch.nn.Module, None, None]:
        """Get iterator over the translators within the lens."""
        yield from self.layer_translators

    def transform_hidden(self, h: torch.Tensor, idx: int) -> torch.Tensor:
        """Transform hidden state from layer `idx`."""
        # Note that we add the translator output residually, in contrast to the formula
        # in the paper. By parametrizing it this way we ensure that weight decay
        # regularizes the transform toward the identity, not the zero transformation.
        return h + self[idx](h)

    def forward(self, h: torch.Tensor, idx: int) -> torch.Tensor:
        """Transform and then decode the hidden states into logits."""
        h = self.transform_hidden(h, idx)
        return self.unembed(h)

    def __len__(self) -> int:
        """Return the number of layer translators in the lens."""
        return len(self.layer_translators)

    @classmethod
    def from_model_and_pretrained_lens(
        cls, model: PreTrainedModel, lens_path: str, bias: bool = True
    ) -> "TunedLens":
        """Create a TunedLens from a pretrained lens checkpoint.

        Args:
            model: The base model to use.
            lens_path: The path to the pretrained lens checkpoint in safetensors format.
        """
        lens = cls(model, bias=bias)
        state_dict = torch_load(lens_path)
        lens.load_state_dict(state_dict, strict=False)
        return lens
