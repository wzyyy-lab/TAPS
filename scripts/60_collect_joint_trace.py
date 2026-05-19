from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dflash import cuda_time
from ddtree import build_ddtree_tree
from joint.config import JointDDTConfig
from joint.lattice import extract_topk_lattice
from joint.pool import CandidateTrie, SOURCE_MARGINAL
from joint.trace import HiddenProvenance, extract_target_child_labels, make_trace_record, save_trace_records
from joint.tree import SelectedTree, compile_joint_tree, follow_tree_tensorized
from model import DFlashDraftModel, extract_context_feature, load_and_process_dataset, sample, select_dataset_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", type=str, required=True)
    parser.add_argument("--draft-model", type=str, required=True)
    parser.add_argument("--datasets", type=str, required=True)
    parser.add_argument("--topk-collect", type=int, default=None)
    parser.add_argument("--candidate-pool-nodes", type=int, default=2048)
    parser.add_argument("--candidate-pool-sequences", type=int, default=256)
    parser.add_argument("--tree-budget-baseline", type=int, default=1024)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--records-per-file", type=int, default=128)
    return parser.parse_args()


def ddtree_tree_to_candidate_trie(
    *,
    draft_logits: torch.Tensor,
    node_token_ids: torch.Tensor,
    node_depths: torch.Tensor,
    parents: list[int],
    tree_budget: int,
) -> CandidateTrie:
    device = draft_logits.device
    token_ids = node_token_ids.to(device=device, dtype=torch.long)
    depths = node_depths.to(device=device, dtype=torch.long)
    parent_tensor = torch.tensor(parents, dtype=torch.long, device=device)
    num_nodes = int(token_ids.numel())
    if num_nodes == 0:
        return CandidateTrie(
            token_ids=token_ids,
            depths=depths,
            parents=parent_tensor,
            ranks=torch.empty(0, dtype=torch.long, device=device),
            step_log_probs=torch.empty(0, dtype=torch.float32, device=device),
            cum_log_probs=torch.empty(0, dtype=torch.float32, device=device),
            source_ids=torch.empty(0, dtype=torch.long, device=device),
            path_hashes=torch.empty(0, dtype=torch.long, device=device),
        )

    rank_k = min(max(int(tree_budget), num_nodes, 1), int(draft_logits.shape[-1]))
    logits = draft_logits.float()
    top_logits, top_token_ids = torch.topk(logits, k=rank_k, dim=-1)
    top_log_probs = top_logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    depth_indices = depths - 1
    per_node_top_tokens = top_token_ids.index_select(0, depth_indices)
    matches = per_node_top_tokens == token_ids.unsqueeze(1)
    if not bool(matches.any(dim=1).all().item()):
        missing = (~matches.any(dim=1)).nonzero(as_tuple=False).flatten()[:8].detach().cpu().tolist()
        raise ValueError(
            "DDTree trace conversion failed: some DDTree nodes were not found in the same-budget top-k list; "
            f"tree_budget={tree_budget}, rank_k={rank_k}, missing_edges={missing}"
        )
    ranks = matches.float().argmax(dim=1).long()
    step_log_probs = top_log_probs[depth_indices, ranks].float()
    cum_log_probs = torch.empty_like(step_log_probs)
    for node_index in range(1, num_nodes + 1):
        edge_index = node_index - 1
        parent_index = int(parent_tensor[node_index].item())
        if parent_index <= 0:
            cum_log_probs[edge_index] = step_log_probs[edge_index]
        else:
            cum_log_probs[edge_index] = cum_log_probs[parent_index - 1] + step_log_probs[edge_index]

    return CandidateTrie(
        token_ids=token_ids,
        depths=depths,
        parents=parent_tensor,
        ranks=ranks,
        step_log_probs=step_log_probs,
        cum_log_probs=cum_log_probs,
        source_ids=torch.full((num_nodes,), SOURCE_MARGINAL, dtype=torch.long, device=device),
        path_hashes=torch.arange(1, num_nodes + 1, dtype=torch.long, device=device),
    )


@torch.inference_mode()
def collect_for_prompt(
    *,
    target,
    draft_model,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    eos_token_id: int,
    config: JointDDTConfig,
    tree_budget_baseline: int,
) -> list[dict]:
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    draft_horizon = block_size - 1
    max_tree_nodes = 1 + int(tree_budget_baseline)
    output_ids = torch.full((1, max_length + max_tree_nodes), mask_token_id, dtype=torch.long, device=draft_model.device)
    position_ids = torch.arange(output_ids.shape[1], device=draft_model.device).unsqueeze(0)
    verify_input_ids_buffer = torch.empty((1, max_tree_nodes), dtype=torch.long, device=draft_model.device)
    verify_position_ids_buffer = torch.empty((1, max_tree_nodes), dtype=torch.long, device=draft_model.device)
    attention_mask_buffer = torch.zeros(
        (1, 1, max_tree_nodes, max_length + max_tree_nodes),
        dtype=target.dtype,
        device=draft_model.device,
    )
    tree_visibility_buffer = torch.empty((max_tree_nodes, max_tree_nodes), dtype=torch.bool, device=draft_model.device)
    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(output.logits, 0.0)
    target_hidden = extract_context_feature(output.hidden_states, draft_model.target_layer_ids)

    hidden_provenance = HiddenProvenance(layer_ids=[int(x) for x in draft_model.target_layer_ids])
    records: list[dict] = []
    start = num_input_tokens
    previous_tree_start = 0
    previous_tree_length = 0

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        root_token = block_output_ids[:, :1]
        noise_embedding = target.model.embed_tokens(block_output_ids)
        draft_logits = target.lm_head(
            draft_model(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[:, past_key_values_draft.get_seq_length() : start + block_size],
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False,
            )[:, -draft_horizon:, :]
        )
        past_key_values_draft.crop(start)

        lattice = extract_topk_lattice(draft_logits[0], config.joint_topk)
        node_token_ids, node_depths, parents, _, visibility_cpu, _ = build_ddtree_tree(
            draft_logits[0], int(tree_budget_baseline)
        )
        trie = ddtree_tree_to_candidate_trie(
            draft_logits=draft_logits[0],
            node_token_ids=node_token_ids,
            node_depths=node_depths,
            parents=parents,
            tree_budget=int(tree_budget_baseline),
        )
        selected_tree = SelectedTree(
            token_ids=trie.token_ids,
            depths=trie.depths,
            parents=trie.parents,
            visibility=visibility_cpu.to(device=trie.device),
            old_to_new=torch.arange(trie.num_total_nodes, dtype=torch.long, device=trie.device),
            selected_old_node_ids=torch.arange(1, trie.num_total_nodes, dtype=torch.long, device=trie.device),
        )
        verify_input_ids, verify_position_ids, verify_attention_mask, previous_tree_start, previous_tree_length = compile_joint_tree(
            root_token_id=root_token[0, 0],
            start=start,
            selected_tree=selected_tree,
            past_length=start,
            dtype=target.dtype,
            verify_input_ids_buffer=verify_input_ids_buffer,
            verify_position_ids_buffer=verify_position_ids_buffer,
            attention_mask_buffer=attention_mask_buffer,
            tree_visibility_buffer=tree_visibility_buffer,
            previous_tree_start=previous_tree_start,
            previous_tree_length=previous_tree_length,
        )
        output = target(
            verify_input_ids,
            position_ids=verify_position_ids,
            attention_mask=verify_attention_mask,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )
        target_labels = extract_target_child_labels(output.logits, trie)
        posterior = sample(output.logits, 0.0)
        accepted_indices, next_token = follow_tree_tensorized(selected_tree, posterior)
        target_greedy_tokens = []
        current_for_oracle = 0
        child_parent_ids = selected_tree.parents[1:]
        child_token_ids = selected_tree.token_ids
        child_indices = torch.arange(1, selected_tree.current_length, dtype=torch.long, device=posterior.device)
        for _ in range(draft_horizon):
            token = int(posterior[0, current_for_oracle].item())
            target_greedy_tokens.append(token)
            matches = (child_parent_ids == current_for_oracle) & (child_token_ids == token)
            match_idx = matches.nonzero(as_tuple=False)
            if match_idx.numel() == 0:
                break
            current_for_oracle = int(child_indices[match_idx[0, 0]].item())
        accepted_index_tensor = torch.tensor(accepted_indices, dtype=torch.long, device=verify_input_ids.device)
        target_hidden_proj = target_hidden[:, -1:, :].detach()
        record = make_trace_record(
            lattice_data={
                "top_token_ids": lattice.top_token_ids,
                "top_log_probs": lattice.top_log_probs,
                "position_entropy": lattice.position_entropy,
                "top1_top2_margin": lattice.top1_top2_margin,
                "topk_mass": lattice.topk_mass,
            },
            trie=trie,
            target_labels=target_labels,
            root_token_id=int(root_token[0, 0].item()),
            round_start=start,
            target_hidden_proj=target_hidden_proj,
            hidden_provenance=hidden_provenance,
            accepted_indices=accepted_indices,
            target_greedy_tokens=target_greedy_tokens,
            ddtree_accept_length=len(accepted_indices),
        )
        record["trace_candidate_source"] = "ddtree_heap"
        record["trace_tree_budget_baseline"] = int(tree_budget_baseline)
        record["trace_candidate_nodes"] = int(trie.num_nodes)
        records.append(record)

        output_ids[:, start : start + len(accepted_indices)] = verify_input_ids.index_select(1, accepted_index_tensor)
        output_ids[:, start + len(accepted_indices)] = next_token
        from ddtree import compact_dynamic_cache

        compact_dynamic_cache(past_key_values_target, start, accepted_index_tensor)
        target_hidden = extract_context_feature(output.hidden_states, draft_model.target_layer_ids).index_select(1, accepted_index_tensor)
        start += len(accepted_indices)
        if eos_token_id is not None:
            new_tokens = output_ids[:, start - len(accepted_indices) : start + 1]
            if torch.isin(new_tokens[0], torch.tensor([eos_token_id], device=output_ids.device)).any():
                break
    return records


def main() -> None:
    args = parse_args()
    if args.temperature >= 1e-5:
        raise ValueError("Trace collection for Joint-DDT v1 only supports temperature=0.0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = AutoModelForCausalLM.from_pretrained(args.target_model, attn_implementation="sdpa", dtype=torch.bfloat16).to(device).eval()
    draft_model = DFlashDraftModel.from_pretrained(args.draft_model, attn_implementation="flash_attention_2", dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    block_size = args.block_size if args.block_size is not None else draft_model.block_size
    topk_collect = args.tree_budget_baseline if args.topk_collect is None else args.topk_collect
    if int(topk_collect) < int(args.tree_budget_baseline):
        raise ValueError("--topk-collect must be >= --tree-budget-baseline for DDTree-budget trace collection")
    config = JointDDTConfig(
        joint_topk=topk_collect,
        candidate_pool_nodes=args.tree_budget_baseline,
        candidate_pool_sequences=args.tree_budget_baseline,
        max_verify_nodes=args.tree_budget_baseline,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    buffer: list[dict] = []
    file_index = 0
    for dataset_name in args.datasets.split(","):
        dataset = load_and_process_dataset(dataset_name.strip())
        dataset = select_dataset_samples(
            dataset,
            max_samples=args.max_samples,
            sample_offset=args.sample_offset,
            shuffle_seed=args.shuffle_seed,
        )
        for instance in tqdm(dataset, desc=f"collect {dataset_name}"):
            messages = []
            for user_content in instance["turns"]:
                messages.append({"role": "user", "content": user_content})
                input_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
                records = collect_for_prompt(
                    target=target,
                    draft_model=draft_model,
                    input_ids=input_ids,
                    mask_token_id=draft_model.mask_token_id,
                    max_new_tokens=args.max_new_tokens,
                    block_size=block_size,
                    eos_token_id=tokenizer.eos_token_id,
                    config=config,
                    tree_budget_baseline=args.tree_budget_baseline,
                )
                buffer.extend(records)
                if len(buffer) >= args.records_per_file:
                    save_trace_records(buffer, output_dir / f"joint_trace_{file_index:05d}.pt")
                    buffer = []
                    file_index += 1
    if buffer:
        save_trace_records(buffer, output_dir / f"joint_trace_{file_index:05d}.pt")
    _ = cuda_time()


if __name__ == "__main__":
    main()
