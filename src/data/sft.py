import os

from datasets import (
    DatasetDict,
    concatenate_datasets,
    get_dataset_config_names,
    load_dataset,
)
from datasets.exceptions import DatasetNotFoundError


def _limit(split, limit: int | None):
    if limit is None:
        return split
    return split.select(range(min(int(limit), len(split))))


def _map_messages(dataset, prompt_fn, response_fn):
    def convert(example):
        return {
            "messages": [
                {"role": "user", "content": prompt_fn(example)},
                {"role": "assistant", "content": response_fn(example)},
            ]
        }

    return dataset.map(
        convert,
        remove_columns=dataset.column_names,
        num_proc=4,
    )


def _load_alpaca(name_or_path: str) -> DatasetDict:
    dataset = load_dataset(name_or_path)["train"]
    dataset = _map_messages(
        dataset,
        lambda row: "\n\n".join(
            part
            for part in [
                (row.get("instruction") or "").strip(),
                (row.get("input") or "").strip(),
            ]
            if part
        ),
        lambda row: (row.get("output") or "").strip(),
    )
    return dataset.train_test_split(test_size=0.1, seed=42)


def _load_gsm8k(name_or_path: str, subset: str = "main") -> DatasetDict:
    dataset = load_dataset(name_or_path, subset)
    return DatasetDict(
        {
            split: _map_messages(
                data,
                lambda row: f"Question: {(row.get('question') or '').strip()}\nAnswer:",
                lambda row: (row.get("answer") or "").strip(),
            )
            for split, data in dataset.items()
        }
    )


def _load_math(name_or_path: str, subset: str | None = None) -> DatasetDict:
    config_names = [subset] if subset else get_dataset_config_names(name_or_path)
    splits: dict[str, list] = {"train": [], "test": []}

    for config_name in config_names:
        dataset = load_dataset(name_or_path, config_name)
        for split in splits:
            if split in dataset:
                splits[split].append(
                    _map_messages(
                        dataset[split],
                        lambda row: (
                            f"Problem:\n{(row.get('problem') or '').strip()}"
                            "\n\nSolution:"
                        ),
                        lambda row: (row.get("solution") or "").strip(),
                    )
                )

    return DatasetDict(
        {
            split: concatenate_datasets(parts)
            for split, parts in splits.items()
            if parts
        }
    )


def _load_self_oss(name_or_path: str) -> DatasetDict:
    dataset = load_dataset(name_or_path)
    train = dataset["train"] if "train" in dataset else next(iter(dataset.values()))
    train = _map_messages(
        train,
        lambda row: (row.get("instruction") or "").strip(),
        lambda row: (row.get("response") or "").strip(),
    )
    split = train.train_test_split(test_size=0.1, seed=42)
    return DatasetDict({"train": split["train"], "test": split["test"]})


def _load_mbpp(name_or_path: str, subset: str | None = None) -> DatasetDict:
    dataset = load_dataset(name_or_path, subset) if subset else load_dataset(name_or_path)
    train = dataset["train"] if "train" in dataset else next(iter(dataset.values()))

    def prompt(row):
        task = (row.get("prompt") or row.get("text") or "").strip()
        sections = [
            "Write a solution to the following problem and make sure that it passes the tests:",
            "```python",
            task,
            "```",
        ]
        tests = []
        if setup := (row.get("test_setup_code") or "").strip():
            tests.append(setup)
        for key in ("test_imports", "test_list"):
            values = row.get(key) or []
            if isinstance(values, list):
                tests.extend(str(value).strip() for value in values if str(value).strip())
        if tests:
            sections.extend(["", "Tests:", "```python", "\n".join(tests), "```"])
        return "\n".join(sections).strip()

    train = _map_messages(
        train,
        prompt,
        lambda row: (row.get("code") or "").strip(),
    )
    return DatasetDict({"train": train})


def _load_one(spec) -> DatasetDict:
    name_or_path = str(spec.name_or_path)
    subset = spec.get("subset")

    if "alpaca" in name_or_path:
        return _load_alpaca(name_or_path)
    if "gsm8k" in name_or_path:
        return _load_gsm8k(name_or_path, subset or "main")
    if "hendrycks_math" in name_or_path:
        return _load_math(name_or_path, subset)
    if (
        "self-oss-instruct-sc2-exec-filter-50k" in name_or_path
        or "self_oss_instruct" in name_or_path
    ):
        return _load_self_oss(name_or_path)
    if "mbpp" in name_or_path:
        return _load_mbpp(name_or_path, subset)
    raise NotImplementedError(f"Unsupported tuned-lens dataset: {name_or_path}")


def load_sft_dataset(cfg) -> DatasetDict:
    """Load and concatenate the SFT mixture used to train the paper's d-Tuned Lens."""
    parts = []
    for spec in cfg.training_datasets:
        try:
            dataset = _load_one(spec)
        except DatasetNotFoundError:
            base_dir = os.environ.get("DATASET_BASE_DIR")
            if not base_dir:
                raise
            local_spec = spec.copy()
            local_spec.name_or_path = os.path.join(base_dir, str(spec.name_or_path))
            dataset = _load_one(local_spec)

        for split, data in dataset.items():
            limit = spec.get(split)
            dataset[split] = _limit(data, limit)
        parts.append(dataset)

    merged = {}
    for split in {name for part in parts for name in part.keys()}:
        split_parts = [part[split] for part in parts if split in part]
        merged[split] = concatenate_datasets(split_parts)
    return DatasetDict(merged)
