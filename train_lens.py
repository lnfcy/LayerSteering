import os

import accelerate
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from transformers import DataCollatorForSeq2Seq, set_seed

from src.data import load_sft_dataset
from src.lens import TunedLens
from src.trainer import TunedLensTrainer
from src.utils import load_pretrained_model, load_tokenizer
from src.utils.training import build_training_arguments, post_process_dataset


@hydra.main(config_path="configs", config_name="train_lens", version_base=None)
def main(cfg: DictConfig) -> None:
    output_dir = HydraConfig.get().runtime.output_dir
    training_args, data_args = build_training_arguments(cfg, output_dir)
    set_seed(int(training_args.seed))

    base_model = load_pretrained_model(cfg)
    lens = TunedLens(base_model)

    tokenizer = load_tokenizer(cfg)

    def tokenize(example: dict) -> dict:
        input_ids = tokenizer.apply_chat_template(
            example["messages"], tokenize=True, add_generation_prompt=False
        )
        prompt_ids = tokenizer.apply_chat_template(
            example["messages"][:-1], tokenize=True, add_generation_prompt=True
        )
        labels = input_ids.copy()
        labels[: len(prompt_ids)] = [-100] * len(prompt_ids)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "prompt_len": len(prompt_ids),
        }

    with accelerate.PartialState().local_main_process_first():
        dataset = load_sft_dataset(cfg)
        dataset = dataset.map(
            tokenize,
            num_proc=data_args.num_proc,
            remove_columns=dataset["train"].column_names,
        )
        dataset = post_process_dataset(dataset, data_args)

    trainer = TunedLensTrainer(
        model=lens,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test"),
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True,
            label_pad_token_id=tokenizer.pad_token_id,
        ),
    )
    trainer.train()
    trainer.save_model(os.path.join(output_dir, "checkpoint-final"))


if __name__ == "__main__":
    main()
