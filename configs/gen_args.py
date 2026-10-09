GEN_LENGTHS = {
    "gsm8k": 256,
    "math-500": 256,
    "humaneval_instruct": 512,
    "mbpp_instruct": 512,
}


def get_generation_args(task: str, model: str) -> dict:
    del model
    return {
        "gen_length": GEN_LENGTHS[task],
        "block_length": 32,
    }
