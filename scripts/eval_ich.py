import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets
import json
import torch
from tqdm import tqdm
from typing import Optional, Dict, List
from functools import partial
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from accelerate import Accelerator
from transformers import HfArgumentParser
from transformers.utils import logging
from torch.utils.data import DataLoader
from memorag import MemoRAG
from longbench.utils import (
    DATASET2CATEGORY, scorer, DATASET2PROMPT, DATASET2MAXNEWTOKENS,
    makedirs, FileLogger, DefaultDataCollator,
)

logger = logging.get_logger(__name__)

ICH_DATASETS = ["ich_skill_flow", "ich_school_compare", "ich_terminology", "ich_lineage"]


@dataclass
class Args:
    gen_model_path: str = field(
        default="mistralai/Mistral-7B-Instruct-v0.2",
    )
    mem_model_path: str = field(
        default="/share/qhj/memorag-qwen2-7b-inst",
    )
    ret_model_path: str = field(
        default="BAAI/bge-m3",
    )
    cache_dir: str = field(
        default="/share/shared_models/",
    )
    access_token: str = field(
        default=None,  # Set your HF token here or via HF_TOKEN env
    )
    eval_data: str = field(
        default="data/ich_benchmark/ich_qa.json",
        metadata={'help': 'Path to ICH benchmark JSON.'}
    )
    output_dir: str = field(
        default="data/results/ich_benchmark/",
        metadata={'help': 'Output directory for results and logs.'}
    )
    result_dir: Optional[str] = field(
        default=None,
        metadata={'help': 'Subdirectory under output_dir for this run.'}
    )
    dataset_names: List[str] = field(
        default_factory=lambda: ICH_DATASETS,
        metadata={'help': 'ICH datasets to evaluate.'}
    )
    max_length: Optional[int] = field(
        default=None,
        metadata={'help': 'Max input length for truncation.'}
    )
    truncate_from_middle: bool = field(
        default=True,
        metadata={'help': 'Truncate from middle instead of end.'}
    )


def process_ich_data(data, indices, tokenizer, max_length=None, truncate_from_middle=True):
    """
    Process ICH benchmark data.
    Each sample: _id, dataset, context, input, answers
    """
    outputs = {'context': [], 'question': [], "dataset": [], "index": [], "length": []}

    for context, question, dataset, index in zip(
        data['context'], data['input'], data['dataset'], indices
    ):
        if max_length is not None:
            tokenized_context = tokenizer.encode(context, add_special_tokens=False)
            if len(tokenized_context) > max_length:
                if truncate_from_middle:
                    half = int(max_length / 2)
                    context = (
                        tokenizer.decode(tokenized_context[:half])
                        + tokenizer.decode(tokenized_context[-half:])
                    )
                else:
                    context = tokenizer.decode(tokenized_context[-max_length:])

        length = len(tokenizer.encode(context))

        outputs["context"].append(context)
        outputs["question"].append(question)
        outputs["dataset"].append(dataset)
        outputs["index"].append(index)
        outputs["length"].append(length)

    return outputs


@torch.no_grad()
def main():
    parser = HfArgumentParser([Args])
    args = parser.parse_args_into_dataclasses()[0]
    accelerator = Accelerator(cpu=False)

    pipe = MemoRAG(
        mem_model_name_or_path=args.mem_model_path,
        ret_model_name_or_path=args.ret_model_path,
        gen_model_name_or_path=args.gen_model_path,
        cache_dir=args.cache_dir,
        access_token=args.access_token,
    )

    tokenizer = pipe.gen_model.tokenizer

    with accelerator.main_process_first():
        process_fn = partial(
            process_ich_data,
            tokenizer=tokenizer,
            max_length=args.max_length,
            truncate_from_middle=args.truncate_from_middle,
        )

        raw_dataset = datasets.load_dataset(
            "json", data_files=args.eval_data, split="train"
        )
        dataset = raw_dataset.map(
            process_fn, batched=True, num_proc=4, with_indices=True,
            remove_columns=raw_dataset.column_names,
        )

    groupby_dataset = dataset.to_pandas().groupby("dataset")

    metrics = {}
    dataset_names = args.dataset_names

    result_dir = os.path.join(args.output_dir, args.result_dir) if args.result_dir else args.output_dir

    for i, dataset_name in enumerate(dataset_names):
        if dataset_name not in groupby_dataset.groups:
            if accelerator.process_index == 0:
                logger.warning(f"Dataset '{dataset_name}' not found in data, skipping.")
            continue

        if accelerator.process_index == 0:
            logger.info(f"Evaluating {dataset_name} ({i + 1} / {len(dataset_names)})...")

        result_path = os.path.join(result_dir, f"{dataset_name}.json")

        dataset_subset = datasets.Dataset.from_pandas(
            groupby_dataset.get_group(dataset_name), preserve_index=False
        )

        data_collator = DefaultDataCollator(padding_side="left")
        dataloader = DataLoader(dataset_subset, batch_size=1, collate_fn=data_collator)
        dataloader = accelerator.prepare(dataloader)

        indices = []
        preds = []
        _prompt = DATASET2PROMPT[dataset_name]
        task_max_new_token = DATASET2MAXNEWTOKENS[dataset_name]

        for x in tqdm(dataloader, desc=f"Generating {dataset_name}"):
            x.pop("dataset")
            index = x.pop("index")[0]

            output = [
                pipe(
                    x["context"][0],
                    x["question"][0],
                    prompt_template=_prompt,
                    task_type="memorag",
                    max_new_tokens=task_max_new_token,
                    reset_each_call=True,
                    use_memory_answer=True,
                )
            ]

            if accelerator.num_processes > 1:
                output = accelerator.gather_for_metrics(output)
                index = accelerator.gather_for_metrics(index)

            if accelerator.process_index == 0:
                preds.extend(output)
                if isinstance(index, (list, torch.Tensor)):
                    if torch.is_tensor(index):
                        index = index.tolist()
                    if isinstance(index, list):
                        indices.extend(index)
                    else:
                        indices.append(index)
                else:
                    indices.append(index)

        if accelerator.process_index == 0:
            raw_dataset_subset = raw_dataset[indices]
            answers = raw_dataset_subset["answers"]
            all_classes = []
            score = scorer(dataset_name, preds, answers, all_classes)

            logger.info(f"{dataset_name}: {score}")
            metrics[dataset_name] = score

            with open(makedirs(result_path), "w", encoding="utf-8") as f:
                f.write(json.dumps(score, ensure_ascii=False) + "\n")
                for idx, pred in zip(indices, preds):
                    sample = raw_dataset[idx]
                    del sample["context"]
                    sample["pred"] = pred
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    if accelerator.process_index == 0:
        category_metrics = defaultdict(list)
        for dataset, metric in metrics.items():
            category = DATASET2CATEGORY[dataset]
            category_metrics[category].append(metric)
        for k, v in category_metrics.items():
            if isinstance(v[0], dict):
                category_metric = {}
                for kk in v[0].keys():
                    vv = [v[j][kk] for j in range(len(v))]
                    category_metric[kk] = round(sum(vv) / len(vv), 2)
                category_metrics[k] = category_metric
            else:
                category_metrics[k] = round(sum(v) / len(v), 2)

        avg = round(sum(metrics.values()) / len(metrics), 2)
        metrics["avg"] = avg

        accelerator.print(metrics)
        with open(os.path.join(args.output_dir, "metrics.jsonl"), "a") as f:
            save_args = asdict(args)
            save_args["metrics"] = metrics
            save_args["category_metrics"] = category_metrics
            f.write(json.dumps(save_args, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
