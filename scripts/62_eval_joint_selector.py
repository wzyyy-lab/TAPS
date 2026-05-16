from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from joint.config import JointDDTConfig
from joint.lattice import TopKLattice
from joint.model import load_node_value_net
from joint.pool import candidate_trie_from_dict
from joint.selector import select_joint_tree
from joint.trace import assert_hidden_provenance_matches, load_trace_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--baseline", type=str, default="ddtree_1024")
    parser.add_argument("--target-accept-drop", type=float, default=0.03)
    parser.add_argument("--max-verify-nodes", type=int, default=192)
    parser.add_argument("--max-verify-sequences", type=int, default=64)
    parser.add_argument("--utility-thresholds", type=str, default="0,0.001,0.003,0.01,0.03,0.1")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def make_lattice(record: dict, device: torch.device) -> TopKLattice:
    return TopKLattice(
        top_token_ids=record["top_token_ids"].to(device=device, dtype=torch.long),
        top_log_probs=record["top_log_probs"].to(device=device, dtype=torch.float32),
        position_entropy=record["position_entropy"].to(device=device, dtype=torch.float32),
        top1_top2_margin=record["top1_top2_margin"].to(device=device, dtype=torch.float32),
        topk_mass=record["topk_mass"].to(device=device, dtype=torch.float32),
        log_z=torch.empty(record["top_token_ids"].shape[0], dtype=torch.float32, device=device),
    )


def accepted_length_from_selected(record: dict, selected_tree) -> int:
    target_next = record["target_next_token_per_parent"].tolist()
    child_map: dict[tuple[int, int], tuple[int, int]] = {}
    parents = selected_tree.parents.detach().cpu().tolist()
    tokens = selected_tree.token_ids.detach().cpu().tolist()
    old_ids = selected_tree.selected_old_node_ids.detach().cpu().tolist()
    for idx, token in enumerate(tokens, start=1):
        child_map[(int(parents[idx]), int(token))] = (idx, int(old_ids[idx - 1]))
    current_new = 0
    current_old = 0
    accepted = 1
    while current_old < len(target_next):
        token = int(target_next[current_old])
        nxt = child_map.get((current_new, token))
        if nxt is None:
            break
        current_new, current_old = nxt
        accepted += 1
    return accepted


def target_topk_oracle_length(record: dict, k: int) -> int:
    target_next = record["target_next_token_per_parent"].tolist()
    trie = record["candidate_trie"]
    parents = [int(x) for x in trie.get("parents", [-1])]
    tokens = [int(x) for x in trie.get("token_ids", [])]
    ranks = [int(x) for x in trie.get("ranks", [])]
    children: dict[int, list[tuple[int, int, int]]] = {}
    for old_node, token in enumerate(tokens, start=1):
        if ranks[old_node - 1] < k:
            children.setdefault(parents[old_node], []).append((int(token), ranks[old_node - 1], old_node))

    memo: dict[int, int] = {}

    def best_from(old_node: int) -> int:
        if old_node in memo:
            return memo[old_node]
        if old_node >= len(target_next):
            memo[old_node] = 1
            return 1
        wanted = int(target_next[old_node])
        best = 1
        for token, _, child in children.get(old_node, []):
            if token == wanted:
                best = max(best, 1 + best_from(child))
        memo[old_node] = best
        return best

    return best_from(0)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    records = load_trace_records(args.traces)
    if not records:
        raise ValueError("No trace records found")

    model, payload = load_node_value_net(args.checkpoint, device=device, dtype=torch.float32)
    hidden_provenance = payload.get("hidden_provenance", {})
    if hidden_provenance and "hidden_provenance" in records[0]:
        assert_hidden_provenance_matches(records[0]["hidden_provenance"], hidden_provenance)

    thresholds = [float(x) for x in args.utility_thresholds.split(",")]
    results = []
    baseline_accept = sum(int(r.get("ddtree_accept_length") or len(r.get("accepted_indices", [])) or 1) for r in records) / len(records)
    oracle_coverage = {}
    for k in (16, 32, 64):
        lengths = [target_topk_oracle_length(record, k) for record in records]
        oracle_accept = sum(lengths) / max(len(lengths), 1)
        oracle_coverage[f"k{k}"] = {
            "oracle_accept": oracle_accept,
            "relative_to_baseline": oracle_accept / max(baseline_accept, 1e-6),
            "draft_topk_limited": oracle_accept < 0.95 * baseline_accept,
        }

    for threshold in thresholds:
        config = JointDDTConfig.from_dict(payload.get("joint_config"))
        config.utility_threshold = threshold
        config.max_verify_nodes = args.max_verify_nodes
        config.max_verify_sequences = args.max_verify_sequences
        accepts = []
        nodes = []
        fallbacks = 0
        for record in tqdm(records, desc=f"eval tau={threshold:g}"):
            trie = candidate_trie_from_dict(record["candidate_trie"], device=device)
            lattice = make_lattice(record, device)
            context_hidden = None
            if model.context_hidden_dim > 0 and "target_hidden_proj" in record:
                hidden = record["target_hidden_proj"].to(device=device, dtype=torch.float32)
                context_hidden = hidden.reshape(-1, hidden.shape[-1])[-1:].contiguous()
            selection = select_joint_tree(
                trie,
                lattice,
                torch.tensor(int(record["root_token_id"]), dtype=torch.long, device=device),
                model,
                config,
                prompt_length=int(record.get("round_start", 0)),
                context_hidden=context_hidden,
                calibration=payload.get("calibration", {}),
            )
            accepts.append(accepted_length_from_selected(record, selection.selected_tree))
            nodes.append(selection.selected_tree.num_nodes)
            fallbacks += int(selection.fallback_reason is not None)
        mean_accept = sum(accepts) / len(accepts)
        mean_nodes = sum(nodes) / len(nodes)
        drop = max(0.0, (baseline_accept - mean_accept) / max(baseline_accept, 1e-6))
        results.append({
            "utility_threshold": threshold,
            "baseline_accept": baseline_accept,
            "joint_accept": mean_accept,
            "accept_drop": drop,
            "mean_nodes": mean_nodes,
            "fallback_rate": fallbacks / len(records),
            "meets_target": drop <= args.target_accept_drop,
        })

    best = sorted(results, key=lambda item: (not item["meets_target"], item["mean_nodes"], item["accept_drop"]))[0]
    tuned_config = payload.get("joint_config", {}) or {}
    tuned_config["utility_threshold"] = best["utility_threshold"]
    tuned_config["max_verify_nodes"] = args.max_verify_nodes
    tuned_config["max_verify_sequences"] = args.max_verify_sequences
    payload_out = {
        "baseline": args.baseline,
        "best": best,
        "results": results,
        "recommended_joint_config": tuned_config,
        "checkpoint_calibration": payload.get("calibration", {}),
        "target_topk_oracle_coverage": oracle_coverage,
    }
    print(json.dumps(payload_out, indent=2))
    if args.output is not None:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload_out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
