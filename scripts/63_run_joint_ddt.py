from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from joint.config import JointDDTConfig
from joint.runtime import joint_ddtree_generate
from model import DFlashDraftModel, load_and_process_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", type=str, required=True)
    parser.add_argument("--draft-model", type=str, required=True)
    parser.add_argument("--joint-checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--joint-topk", type=int, default=32)
    parser.add_argument("--candidate-pool-nodes", type=int, default=2048)
    parser.add_argument("--candidate-pool-sequences", type=int, default=256)
    parser.add_argument("--candidate-pool-source", type=str, default="union", choices=["union", "ddtree_heap"])
    parser.add_argument("--max-verify-sequences", type=int, default=64)
    parser.add_argument("--max-verify-nodes", type=int, default=192)
    parser.add_argument("--fallback-to-ddtree", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fallback-backend", type=str, default="gpu_marginal", choices=["gpu_marginal", "cpu_ddtree", "none"])
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prompt is None and args.dataset is None:
        raise ValueError("Provide either --prompt or --dataset")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = AutoModelForCausalLM.from_pretrained(args.target_model, attn_implementation="sdpa", dtype=torch.bfloat16).to(device).eval()
    draft_model = DFlashDraftModel.from_pretrained(args.draft_model, attn_implementation="flash_attention_2", dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    block_size = args.block_size if args.block_size is not None else draft_model.block_size
    config = JointDDTConfig(
        joint_topk=args.joint_topk,
        candidate_pool_nodes=args.candidate_pool_nodes,
        candidate_pool_sequences=args.candidate_pool_sequences,
        candidate_pool_source=args.candidate_pool_source,
        max_verify_sequences=args.max_verify_sequences,
        max_verify_nodes=args.max_verify_nodes,
        fallback_to_ddtree=args.fallback_to_ddtree,
        fallback_backend=args.fallback_backend,
    )

    prompts: list[str] = []
    if args.prompt is not None:
        prompts.append(args.prompt)
    if args.dataset is not None:
        dataset = load_and_process_dataset(args.dataset)
        if args.max_samples is not None and len(dataset) > args.max_samples:
            dataset = dataset.select(range(args.max_samples))
        prompts.extend(instance["turns"][0] for instance in dataset)

    for prompt in prompts:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
        response = joint_ddtree_generate(
            model=draft_model,
            target=target,
            input_ids=input_ids,
            mask_token_id=draft_model.mask_token_id,
            max_new_tokens=args.max_new_tokens,
            block_size=block_size,
            stop_token_ids=[tokenizer.eos_token_id],
            temperature=args.temperature,
            joint_checkpoint=args.joint_checkpoint,
            joint_config=config,
            save_tree_traces=True,
        )
        generated = response.output_ids[0, response.num_input_tokens :]
        print(tokenizer.decode(generated, skip_special_tokens=True))
        print({
            "time_per_output_token": response.time_per_output_token,
            "acceptance_lengths": response.acceptance_lengths,
            "stage_times": response.stage_times,
            "fallback_counts": response.fallback_counts,
        })


if __name__ == "__main__":
    main()
