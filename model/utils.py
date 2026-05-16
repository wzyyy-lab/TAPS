import os
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset, Features, Sequence, Value


_DEFAULT_HF_ASSETS = Path("hf_assets")


def _hf_assets_dir() -> Path:
    return Path(os.environ.get("TAPS_HF_ASSETS", _DEFAULT_HF_ASSETS))


def _local_dataset_file(*parts: str) -> Path | None:
    path = _hf_assets_dir().joinpath("datasets", *parts)
    return path if path.exists() else None


def select_dataset_samples(dataset, max_samples: int | None = None, sample_offset: int = 0, shuffle_seed: int = 0):
    if sample_offset < 0:
        raise ValueError("sample_offset must be non-negative")
    if max_samples is not None and max_samples < 0:
        raise ValueError("max_samples must be non-negative or None")
    if max_samples is None and sample_offset == 0:
        return dataset

    shuffled = dataset.shuffle(seed=shuffle_seed)
    if max_samples is None:
        end = len(shuffled)
    else:
        end = min(len(shuffled), sample_offset + max_samples)
    if sample_offset >= end:
        return shuffled.select([])
    return shuffled.select(range(sample_offset, end))

def build_target_layer_ids(num_target_layers: int, num_draft_layers: int):
    if num_draft_layers == 1:
        return [(num_target_layers // 2)]
    start = 1
    end = num_target_layers - 3
    span = end - start
    target_layer_ids = [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]
    return target_layer_ids

def extract_context_feature(
    hidden_states: list[torch.Tensor],
    layer_ids: Optional[list[int]],
) -> torch.Tensor:
    offset = 1
    selected_states = []
    for layer_id in layer_ids:
        selected_states.append(hidden_states[layer_id + offset])
    target_hidden = torch.cat(selected_states, dim=-1)
    return target_hidden

def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size)
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)

def load_and_process_dataset(data_name: str):
    # Math datasets
    if data_name == "gsm8k":
        local_path = _local_dataset_file("gsm8k", "main", "test-00000-of-00001.parquet")
        if local_path is not None:
            dataset = load_dataset("parquet", data_files={"test": str(local_path)}, split="test")
        else:
            dataset = load_dataset("openai/gsm8k", "main", split="test")
        prompt_fmt = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})
    
    elif data_name == "math500":
        local_path = _local_dataset_file("MATH-500", "test.jsonl")
        if local_path is not None:
            dataset = load_dataset("json", data_files={"test": str(local_path)}, split="test")
        else:
            dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})
    
    elif data_name == "aime24":
        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "aime25":
        local_path = _local_dataset_file("aime_2025", "data", "train-00000-of-00001.parquet")
        if local_path is not None:
            dataset = load_dataset("parquet", data_files={"train": str(local_path)}, split="train")
        else:
            dataset = load_dataset("MathArena/aime_2025", split="train")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    # Chat datasets 
    elif data_name == "alpaca":
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
        dataset = dataset.map(lambda x: {"formatted_input": (f"{x['instruction']}\n\nInput:\n{x['input']}" if x['input'] else x['instruction'])})
        dataset = dataset.map(lambda x: {"turns": [x["formatted_input"]]})

    elif data_name == "mt-bench":
        local_dir = _hf_assets_dir().joinpath("datasets", "mt_bench_prompts", "data")
        local_files = sorted(local_dir.glob("train-*.parquet")) if local_dir.exists() else []
        if local_files:
            dataset = load_dataset("parquet", data_files={"train": [str(path) for path in local_files]}, split="train")
        else:
            dataset = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        dataset = dataset.map(lambda x: {"turns": x["prompt"]})

    # Coding datasets
    elif data_name == "humaneval":
        local_path = _local_dataset_file("openai_humaneval", "openai_humaneval", "test-00000-of-00001.parquet")
        if local_path is not None:
            dataset = load_dataset("parquet", data_files={"test": str(local_path)}, split="test")
        else:
            dataset = load_dataset("openai/openai_humaneval", split="test")
        prompt_fmt = "Write a solution to the following problem and make sure that it passes the tests:\n```python\n{prompt}\n```"
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})
    
    elif data_name == "mbpp":
        local_path = _local_dataset_file("mbpp", "sanitized", "test-00000-of-00001.parquet")
        if local_path is not None:
            dataset = load_dataset("parquet", data_files={"test": str(local_path)}, split="test")
        else:
            dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        dataset = dataset.map(lambda x: {"turns": [x["prompt"]]})
    
    elif data_name == "lbpp":
        LBPP_PY_TEST_URL = "https://huggingface.co/datasets/CohereLabs/lbpp/resolve/main/python/test.parquet"
        dataset = load_dataset("parquet", data_files={"test": LBPP_PY_TEST_URL})["test"]
        dataset = dataset.map(lambda x: {"turns": [x["instruction"]]})

    elif data_name == "swe-bench":
        dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        prompt_fmt = "Problem Statement:\n{problem_statement}\nPlease fix the issue described above."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})
    
    elif data_name == "livecodebench":
        prompt_cache = _local_dataset_file("code_generation_lite", "prompts.jsonl")
        if prompt_cache is not None:
            dataset = load_dataset("json", data_files={"test": str(prompt_cache)}, split="test")
            return dataset
        allowed_files = ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"]
        local_files = [_local_dataset_file("code_generation_lite", filename) for filename in allowed_files]
        if all(path is not None for path in local_files):
            data_files = [str(path) for path in local_files if path is not None]
        else:
            base = "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/"
            data_files = [base + filename for filename in allowed_files]
        dataset = load_dataset("json", data_files={"test": data_files})["test"]
        def format_lcb(doc):
            system_prompt = (
                "You are an expert Python programmer. You will be given a question (problem specification) "
                "and will generate a correct Python program that matches the specification and passes all tests. "
                "You will NOT return anything except for the program"
            )
            question_block = f"### Question:\n{doc['question_content']}"
            if doc.get("starter_code"):
                format_message = "### Format: Use the following code structure:"
                code_block = f"```python\n{doc['starter_code']}\n```"
            else:
                format_message = "### Format: Write your code in the following format:"
                code_block = "```python\n# YOUR CODE HERE\n```"
            answer_footer = "### Answer: (use the provided format with backticks)"
            return f"{system_prompt}\n\n{question_block}\n\n{format_message}\n{code_block}\n\n{answer_footer}"
        target_features = Features({"turns": Sequence(Value("large_string"))})
        dataset = dataset.map(
            lambda x: {"turns": [format_lcb(x)]},
            remove_columns=dataset.column_names,
            features=target_features
        )
    
    return dataset