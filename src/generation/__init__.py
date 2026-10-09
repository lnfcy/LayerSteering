from src.generation.steering import steering_generate
from src.generation.utils import decode_final_frame


def generate(model, input_ids, *, strategy: str, **kwargs):
    if strategy != "steering":
        raise ValueError(f"Unsupported generation strategy: {strategy}")
    return steering_generate(model, input_ids, **kwargs)


__all__ = ["generate", "decode_final_frame"]
