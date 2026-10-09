import os
import shutil
import sys
import tempfile
import warnings
import zipfile
from contextlib import contextmanager
from pathlib import Path

import omegaconf
from accelerate.utils import set_seed
from dill import PickleWarning
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from configs.gen_args import get_generation_args
from .common import LoggerFilter, Timer, apply_fn, tensor_delete, tensor_insert
from .models import load_eval_model, load_pretrained_model, load_tokenizer


@contextmanager
def sympy_antlr_patcher(target_version: str = "4.11.0"):
    repo_root = Path(__file__).resolve().parents[2]
    wheel = next(
        (repo_root / "vendor" / "wheels").glob(
            f"antlr4_python3_runtime-{target_version}-*.whl"
        )
    )
    temp_dir = Path(tempfile.mkdtemp(prefix="antlr4_"))
    original_path = sys.path[:]
    original_modules = {
        name: module
        for name, module in sys.modules.items()
        if name.startswith("antlr4")
    }

    try:
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(temp_dir)
        for name in list(sys.modules):
            if name.startswith("antlr4"):
                del sys.modules[name]
        sys.path.insert(0, str(temp_dir))
        yield
    finally:
        sys.path[:] = original_path
        for name in list(sys.modules):
            if name.startswith("antlr4"):
                del sys.modules[name]
        sys.modules.update(original_modules)
        shutil.rmtree(temp_dir)


def pre_initialize(cfg: DictConfig) -> None:
    load_dotenv()
    set_seed(cfg.seed)
    logger.remove()
    logger.add(sys.stderr, filter=LoggerFilter())
    warnings.filterwarnings("ignore", category=PickleWarning)

    generation_defaults = get_generation_args(cfg.dataset.name, cfg.model.name)
    with omegaconf.open_dict(cfg.generation):
        for key, value in generation_defaults.items():
            if cfg.generation.get(key) is None:
                cfg.generation[key] = value
        for key, value in cfg.model.generation.items():
            if cfg.generation.get(key) is None:
                cfg.generation[key] = value
        OmegaConf.resolve(cfg.generation)

    logger.info("{}", cfg.generation, rank_zero_only=True)
    os.environ["MASK_TOKEN_ID"] = str(cfg.generation.mask_token_id)
    os.environ["EOS_TOKEN_ID"] = str(cfg.generation.eos_token_id)
    os.environ["PAD_TOKEN_ID"] = str(cfg.generation.pad_token_id)


__all__ = [
    "LoggerFilter",
    "Timer",
    "apply_fn",
    "load_eval_model",
    "load_pretrained_model",
    "load_tokenizer",
    "pre_initialize",
    "sympy_antlr_patcher",
    "tensor_delete",
    "tensor_insert",
]
