import argparse
import random
import time
from itertools import chain
from pathlib import Path

from loguru import logger
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import distributed as dist
from model import DFlashDraftModel, load_and_process_dataset, select_dataset_samples
from dflash import dflash_generate
from ddtree import ddtree_generate, maybe_enable_cpp_compact
from joint import JointDDTConfig, joint_ddtree_generate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--draft-name-or-path", type=str, required=True)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--tree-budget", type=str, default="16,32,64,128,256,512,1024")
    parser.add_argument("--proposal-mode", type=str, default="all", choices=["dflash", "ddtree", "joint", "ddtree_joint", "all"])
    parser.add_argument("--joint-checkpoint", type=str, default=None)
    parser.add_argument("--joint-topk", type=int, default=32)
    parser.add_argument("--candidate-pool-nodes", type=int, default=2048)
    parser.add_argument("--candidate-pool-sequences", type=int, default=256)
    parser.add_argument("--candidate-pool-source", type=str, default="union", choices=["union", "ddtree_heap"])
    parser.add_argument("--min-verify-nodes", type=int, default=16)
    parser.add_argument("--max-verify-nodes", type=int, default=192)
    parser.add_argument("--min-verify-sequences", type=int, default=4)
    parser.add_argument("--max-verify-sequences", type=int, default=64)
    parser.add_argument("--fallback-to-ddtree", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fallback-backend", type=str, default="gpu_marginal", choices=["gpu_marginal", "cpu_ddtree", "none"])
    parser.add_argument("--utility-threshold", type=float, default=0.0)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--flash-attn", action="store_true")
    parser.add_argument("--disable-cpp-compact-cache", action="store_true")
    parser.add_argument("--save-path", type=str, default=None)
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dist.init()
    torch.cuda.set_device(dist.local_rank())
    device = torch.device(f"cuda:{dist.local_rank()}")
    maybe_enable_cpp_compact(not args.disable_cpp_compact_cache)

    def has_flash_attn() -> bool:
        try:
            import flash_attn  # noqa: F401
            return True
        except ImportError:
            return False

    installed_flash_attn = has_flash_attn()
    if not installed_flash_attn:
        raise RuntimeError("flash_attn must be installed because the draft DFlash model always uses FlashAttention")

    target_attn_implementation = "flash_attention_2" if args.flash_attn else "sdpa"
    draft_attn_implementation = "flash_attention_2"

    if not args.flash_attn and installed_flash_attn:
        logger.warning("DDTree uses a custom tree attention mask on the target model. For compatibility, forcing the target verifier to torch.sdpa.")

    draft_model = DFlashDraftModel.from_pretrained(
        args.draft_name_or_path,
        attn_implementation=draft_attn_implementation,
        dtype=torch.bfloat16,
    ).to(device).eval()

    block_size = args.block_size if args.block_size is not None else draft_model.block_size
    tree_budgets = [int(tree_budget) for tree_budget in args.tree_budget.split(",")]
    methods_to_run = []
    method_key_to_tree_budget = {}
    if args.proposal_mode in {"dflash", "all"}:
        methods_to_run.append("dflash")
    if args.proposal_mode in {"ddtree", "ddtree_joint", "all"} and not args.flash_attn:
        ddtree_method_keys = [f"ddtree_tb{tree_budget}" for tree_budget in tree_budgets]
        methods_to_run.extend(ddtree_method_keys)
        method_key_to_tree_budget.update({f"ddtree_tb{tree_budget}": tree_budget for tree_budget in tree_budgets})
    if args.proposal_mode in {"joint", "ddtree_joint"}:
        if args.joint_checkpoint is None:
            raise ValueError("--joint-checkpoint is required when --proposal-mode includes joint")
        if args.flash_attn:
            logger.warning("Joint-DDT uses custom tree masks; forcing the target verifier to torch.sdpa.")
        methods_to_run.append("joint")
    elif args.proposal_mode == "all" and args.joint_checkpoint is not None:
        if args.flash_attn:
            logger.warning("Joint-DDT uses custom tree masks; forcing the target verifier to torch.sdpa.")
        methods_to_run.append("joint")
    if not methods_to_run:
        raise ValueError("No proposal methods selected. Disable --flash-attn for DDTree/Joint tree verification or choose dflash.")
    if "joint" in methods_to_run:
        target_attn_implementation = "sdpa"

    joint_config = JointDDTConfig(
        joint_topk=args.joint_topk,
        candidate_pool_nodes=args.candidate_pool_nodes,
        candidate_pool_sequences=args.candidate_pool_sequences,
        candidate_pool_source=args.candidate_pool_source,
        min_verify_nodes=args.min_verify_nodes,
        max_verify_nodes=args.max_verify_nodes,
        min_verify_sequences=args.min_verify_sequences,
        max_verify_sequences=args.max_verify_sequences,
        fallback_to_ddtree=args.fallback_to_ddtree,
        fallback_backend=args.fallback_backend,
        utility_threshold=args.utility_threshold,
    )

    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation=target_attn_implementation,
        dtype=torch.bfloat16,
    ).to(device).eval()


    def timed_generate(fn, *args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        out = fn(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        setattr(out, "wall_time_seconds", time.perf_counter() - start)
        return out

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = load_and_process_dataset(args.dataset)

    dataset = select_dataset_samples(
        dataset,
        max_samples=args.max_samples,
        sample_offset=args.sample_offset,
        shuffle_seed=args.shuffle_seed,
    )

    warmup_input_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Warmup"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    warmup_input_ids = tokenizer.encode(warmup_input_text, return_tensors="pt").to(target.device)
    warmup_max_new_tokens = min(args.max_new_tokens, 16)

    _ = dflash_generate(
        model=draft_model,
        target=target,
        input_ids=warmup_input_ids,
        mask_token_id=draft_model.mask_token_id,
        max_new_tokens=warmup_max_new_tokens,
        block_size=1,
        stop_token_ids=[tokenizer.eos_token_id],
        temperature=args.temperature,
    )
    for method_key in methods_to_run:
        if method_key == "dflash":
            _ = dflash_generate(
                model=draft_model,
                target=target,
                input_ids=warmup_input_ids,
                mask_token_id=draft_model.mask_token_id,
                max_new_tokens=warmup_max_new_tokens,
                block_size=block_size,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
            )
        elif method_key.startswith("ddtree_tb"):
            _ = ddtree_generate(
                model=draft_model,
                target=target,
                input_ids=warmup_input_ids,
                mask_token_id=draft_model.mask_token_id,
                max_new_tokens=warmup_max_new_tokens,
                block_size=block_size,
                tree_budget=method_key_to_tree_budget[method_key],
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
            )
        elif method_key == "joint":
            _ = joint_ddtree_generate(
                model=draft_model,
                target=target,
                input_ids=warmup_input_ids,
                mask_token_id=draft_model.mask_token_id,
                max_new_tokens=warmup_max_new_tokens,
                block_size=block_size,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
                joint_checkpoint=args.joint_checkpoint,
                joint_config=joint_config,
            )

    responses = []
    indices = range(dist.rank(), len(dataset), dist.size())
    for idx in tqdm(indices, disable=not dist.is_main()):
        instance = dataset[idx]
        messages = []
        for user_content in instance["turns"]:
            messages.append({"role": "user", "content": user_content})
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(target.device)

            response = {}
            response["baseline"] = timed_generate(
                dflash_generate,
                model=draft_model,
                target=target,
                input_ids=input_ids,
                mask_token_id=draft_model.mask_token_id,
                max_new_tokens=args.max_new_tokens,
                block_size=1,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
            )
            for method_key in methods_to_run:
                if method_key == "dflash":
                    response[method_key] = timed_generate(
                        dflash_generate,
                        model=draft_model,
                        target=target,
                        input_ids=input_ids,
                        mask_token_id=draft_model.mask_token_id,
                        max_new_tokens=args.max_new_tokens,
                        block_size=block_size,
                        stop_token_ids=[tokenizer.eos_token_id],
                        temperature=args.temperature,
                    )
                elif method_key.startswith("ddtree_tb"):
                    response[method_key] = timed_generate(
                        ddtree_generate,
                        model=draft_model,
                        target=target,
                        input_ids=input_ids,
                        mask_token_id=draft_model.mask_token_id,
                        max_new_tokens=args.max_new_tokens,
                        block_size=block_size,
                        tree_budget=method_key_to_tree_budget[method_key],
                        stop_token_ids=[tokenizer.eos_token_id],
                        temperature=args.temperature,
                    )
                elif method_key == "joint":
                    response[method_key] = timed_generate(
                        joint_ddtree_generate,
                        model=draft_model,
                        target=target,
                        input_ids=input_ids,
                        mask_token_id=draft_model.mask_token_id,
                        max_new_tokens=args.max_new_tokens,
                        block_size=block_size,
                        stop_token_ids=[tokenizer.eos_token_id],
                        temperature=args.temperature,
                        joint_checkpoint=args.joint_checkpoint,
                        joint_config=joint_config,
                    )

            spec_response = response[methods_to_run[-1]]
            generated_ids = spec_response.output_ids[0, spec_response.num_input_tokens :]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": output_text})
            responses.append(response)

    if dist.size() > 1:
        responses = dist.gather(responses, dst=0)
        if not dist.is_main():
            return
        responses = list(chain(*responses))

    run_data = {
        "responses": responses,
        "block_size": block_size,
        "draft_attn_implementation": draft_attn_implementation,
        "target_attn_implementation": target_attn_implementation,
        "args": vars(args),
    }
    
    if args.save_path is not None:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(run_data, save_path)


if __name__ == "__main__":
    main()
