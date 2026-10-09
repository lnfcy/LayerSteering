from dataclasses import dataclass, field

from datasets import DatasetDict
from hydra.utils import instantiate
from transformers import TrainingArguments


@dataclass
class DataArguments:
    num_proc: int = 8
    max_length: int = 2048
    truncation: str = "filter"


@dataclass
class LensTrainingArguments(TrainingArguments):
    output_dir: str | None = None
    decay_ratio: float | None = field(default=None)


def build_training_arguments(cfg, output_dir: str):
    data_args = instantiate(cfg.data_args)
    training_args = instantiate(cfg.training_args)
    training_args.output_dir = output_dir
    training_args.logging_dir = f"{output_dir}/logs"
    return training_args, data_args


def post_process_dataset(dataset: DatasetDict, data_args: DataArguments) -> DatasetDict:
    if data_args.truncation == "filter":
        return dataset.filter(
            lambda row: len(row["input_ids"]) <= data_args.max_length,
            num_proc=data_args.num_proc,
        )
    if data_args.truncation == "right":
        return dataset.map(
            lambda row: {
                **row,
                "input_ids": row["input_ids"][: data_args.max_length],
                "labels": row["labels"][: data_args.max_length],
            },
            num_proc=data_args.num_proc,
        )
    raise ValueError(f"Unknown truncation mode: {data_args.truncation}")
