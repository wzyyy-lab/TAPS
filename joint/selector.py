from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .config import JointDDTConfig
from .calibration import apply_logit_calibration
from .lattice import TopKLattice
from .model import EdgeFeatureBatch, NodeValueNet
from .pool import CandidateTrie
from .segments import grouped_softmax
from .tree import SelectedTree, compact_selected_trie


SCALAR_FEATURE_NAMES = (
    "step_log_prob",
    "cum_log_prob",
    "parent_cum_log_prob",
    "depth",
    "depth_norm",
    "rank",
    "rank_norm",
    "position_entropy",
    "top1_top2_margin",
    "topk_mass",
    "path_mean_log_prob",
    "is_first_depth",
)


@dataclass
class JointSelectionResult:
    selected_tree: SelectedTree
    selected_mask: torch.Tensor
    q_cond: torch.Tensor
    q_other: torch.Tensor
    q_reach: torch.Tensor
    edge_logits: torch.Tensor
    other_logits: torch.Tensor
    metrics: dict[str, float | int | str] = field(default_factory=dict)
    fallback_reason: str | None = None


@dataclass
class EdgeScoreResult:
    edge_logits: torch.Tensor
    other_logits: torch.Tensor
    q_cond: torch.Tensor
    q_other: torch.Tensor
    q_reach: torch.Tensor


def _node_token_ids_with_root(trie: CandidateTrie, root_token_id: torch.Tensor) -> torch.Tensor:
    root = root_token_id.reshape(1).to(device=trie.device, dtype=torch.long)
    return torch.cat([root, trie.token_ids], dim=0)


def build_edge_feature_batch(
    trie: CandidateTrie,
    lattice: TopKLattice,
    root_token_id: torch.Tensor,
    model: NodeValueNet,
    context_hidden: torch.Tensor | None = None,
) -> EdgeFeatureBatch:
    device = trie.device
    edge_count = trie.num_nodes
    total_nodes = trie.num_total_nodes
    scalar_dim = model.scalar_feature_dim

    if edge_count == 0:
        empty_long = torch.empty(0, dtype=torch.long, device=device)
        empty_scalar = torch.empty(0, scalar_dim, dtype=torch.float32, device=device)
        parent_scalar = torch.zeros((1, scalar_dim), dtype=torch.float32, device=device)
        root = root_token_id.reshape(1).to(device=device, dtype=torch.long)
        return EdgeFeatureBatch(
            child_token_ids=empty_long,
            parent_token_ids=empty_long,
            root_token_ids=empty_long,
            depths=empty_long,
            ranks=empty_long,
            source_ids=empty_long,
            scalar_features=empty_scalar,
            parent_token_ids_for_other=root,
            root_token_ids_for_other=root,
            parent_depths=torch.zeros(1, dtype=torch.long, device=device),
            parent_scalar_features=parent_scalar,
            context_hidden=context_hidden,
            parent_context_hidden=context_hidden,
        )

    parent_ids = trie.edge_parent_ids
    all_token_ids = _node_token_ids_with_root(trie, root_token_id)
    parent_token_ids = all_token_ids.gather(0, parent_ids)
    root_ids = root_token_id.reshape(1).to(device=device, dtype=torch.long).expand(edge_count)

    parent_cum = torch.zeros((total_nodes,), dtype=torch.float32, device=device)
    parent_cum[1:] = trie.cum_log_probs
    parent_depths_all = torch.zeros((total_nodes,), dtype=torch.long, device=device)
    parent_depths_all[1:] = trie.depths
    edge_parent_cum = parent_cum.gather(0, parent_ids)

    depth_index = (trie.depths - 1).clamp(0, lattice.horizon - 1)
    depth_float = trie.depths.float()
    rank_float = trie.ranks.float()
    horizon = max(lattice.horizon, 1)
    topk = max(lattice.topk - 1, 1)

    scalar_columns = [
        trie.step_log_probs,
        trie.cum_log_probs,
        edge_parent_cum,
        depth_float,
        depth_float / float(horizon),
        rank_float,
        rank_float / float(topk),
        lattice.position_entropy.gather(0, depth_index).float(),
        lattice.top1_top2_margin.gather(0, depth_index).float(),
        lattice.topk_mass.gather(0, depth_index).float(),
        trie.cum_log_probs / depth_float.clamp_min(1.0),
        (trie.depths == 1).float(),
    ]
    scalar_features = torch.stack(scalar_columns, dim=-1)
    if scalar_features.shape[-1] != scalar_dim:
        raise ValueError(f"NodeValueNet expects {scalar_dim} scalar features, got {scalar_features.shape[-1]}")

    parent_scalar = torch.zeros((total_nodes, scalar_dim), dtype=torch.float32, device=device)
    parent_scalar[1:, 1] = trie.cum_log_probs
    parent_scalar[1:, 2] = parent_cum[trie.parents[1:]]
    parent_scalar[:, 3] = parent_depths_all.float()
    parent_scalar[:, 4] = parent_depths_all.float() / float(horizon)
    parent_scalar[1:, 10] = trie.cum_log_probs / trie.depths.float().clamp_min(1.0)
    parent_scalar[1:, 11] = (trie.depths == 1).float()

    root_ids_for_other = root_token_id.reshape(1).to(device=device, dtype=torch.long).expand(total_nodes)
    return EdgeFeatureBatch(
        child_token_ids=trie.token_ids,
        parent_token_ids=parent_token_ids,
        root_token_ids=root_ids,
        depths=trie.depths,
        ranks=trie.ranks,
        source_ids=trie.source_ids,
        scalar_features=scalar_features,
        parent_token_ids_for_other=all_token_ids,
        root_token_ids_for_other=root_ids_for_other,
        parent_depths=parent_depths_all,
        parent_scalar_features=parent_scalar,
        context_hidden=context_hidden,
        parent_context_hidden=context_hidden,
    )


def propagate_reach(trie: CandidateTrie, q_cond: torch.Tensor, max_depth: int) -> torch.Tensor:
    """Propagate acceptance probability from root through the trie.

    Fully GPU — zero CPU↔GPU synchronization.  Uses iterative relaxation:
    each iteration correctly resolves one more depth level, so after
    *max_depth* iterations every node has its final reach value.
    """
    q_reach = torch.zeros((trie.num_total_nodes,), dtype=q_cond.dtype, device=trie.device)
    q_reach[0] = 1.0
    if trie.num_nodes == 0:
        return q_reach
    parents_nonroot = trie.parents[1:]
    for _ in range(max_depth):
        q_reach[1:] = q_reach.gather(0, parents_nonroot) * q_cond
    q_reach.clamp_(max=1.0)
    return q_reach


def build_ancestor_matrix(parents: torch.Tensor, max_depth: int | None = None) -> torch.Tensor:
    total_nodes = int(parents.numel())
    if total_nodes == 0:
        raise ValueError("parents must include root")
    if max_depth is None:
        max_depth = total_nodes
    max_depth = max(1, int(max_depth))
    parent_for_gather = parents.clamp_min(0)
    current = torch.arange(total_nodes, dtype=torch.long, device=parents.device)
    columns = [current]
    for _ in range(max_depth):
        current = parent_for_gather.gather(0, current)
        columns.append(current)
    return torch.stack(columns, dim=1)


def _closure_mask_from_ranked_nodes(
    ranked_old_node_ids: torch.Tensor,
    parents: torch.Tensor,
    max_nonroot_nodes: int,
    min_nonroot_nodes: int,
    max_depth: int,
    max_endpoints: int | None = None,
) -> torch.Tensor:
    device = parents.device
    total_nodes = int(parents.numel())
    selected = torch.zeros((total_nodes,), dtype=torch.bool, device=device)
    selected[0] = True
    endpoint_limit = int(ranked_old_node_ids.numel() if max_endpoints is None else max(0, max_endpoints))
    if ranked_old_node_ids.numel() == 0 or max_nonroot_nodes <= 0 or endpoint_limit <= 0:
        return selected

    # --- GPU: vectorized ancestor chain computation ---
    # Walk parent chains for ALL ranked nodes simultaneously via gather,
    # replacing per-node Python while-loop + parents.cpu().tolist().
    parent_for_gather = parents.clamp_min(0)
    current = ranked_old_node_ids
    chains = [current]
    for _ in range(max_depth):
        current = parent_for_gather.gather(0, current)
        chains.append(current)
    ancestor_matrix = torch.stack(chains, dim=1)  # [num_ranked, max_depth+1]

    # --- Single bulk transfer to CPU ---
    ancestors_cpu = ancestor_matrix.cpu().tolist()

    # --- CPU: build closures from ancestor rows ---
    ancestor_closures: list[set[int]] = []
    for row in ancestors_cpu:
        ancestor_closures.append({x for x in row if x != 0})

    # --- CPU: greedy selection (sequential, optimal for this size) ---
    selected_set: set[int] = set()
    endpoints = 0
    for idx in range(len(ancestor_closures)):
        closure = ancestor_closures[idx]
        candidate = selected_set | closure
        count = len(candidate)
        if count > max_nonroot_nodes or count == len(selected_set):
            continue
        selected_set = candidate
        endpoints += 1
        if count >= max_nonroot_nodes or endpoints >= endpoint_limit:
            break

    if len(selected_set) < min_nonroot_nodes:
        for idx in range(len(ancestor_closures)):
            closure = ancestor_closures[idx]
            candidate = selected_set | closure
            count = len(candidate)
            if count <= max_nonroot_nodes:
                selected_set = candidate
            if len(selected_set) >= min_nonroot_nodes:
                break

    # --- Single transfer back to GPU ---
    if selected_set:
        idx_tensor = torch.tensor(sorted(selected_set), dtype=torch.long, device=device)
        selected[idx_tensor] = True
    return selected


def _leaf_count(parents: torch.Tensor) -> int:
    if parents.numel() <= 1:
        return 0
    child_counts = torch.zeros((parents.numel(),), dtype=torch.long, device=parents.device)
    child_counts.scatter_add_(0, parents[1:], torch.ones_like(parents[1:]))
    return int((child_counts[1:] == 0).sum().item())


def _select_prefix_closed(
    trie: CandidateTrie,
    q_reach: torch.Tensor,
    config: JointDDTConfig,
    prompt_length: int,
    max_depth: int,
) -> torch.Tensor:
    device = trie.device
    selected = torch.zeros((trie.num_total_nodes,), dtype=torch.bool, device=device)
    selected[0] = True
    if trie.num_nodes == 0 or config.max_verify_nodes <= 0:
        return selected[1:]

    q_nodes = q_reach[1:]
    depth_cost = torch.clamp(
        float(config.lambda_node)
        + float(config.lambda_prompt) * float(prompt_length)
        + 2.0 * float(config.lambda_quadratic) * trie.depths.float().clamp_min(1.0),
        min=1e-6,
    )
    utility = q_nodes / depth_cost
    max_nodes = int(config.max_verify_nodes)
    min_nodes = min(int(config.min_verify_nodes), trie.num_nodes, max_nodes)
    valid = trie.ranks < int(config.per_parent_child_cap)
    if config.utility_threshold > 0:
        valid &= utility >= float(config.utility_threshold)
    if not valid.any():
        valid = torch.ones_like(valid)
    valid_indices = valid.nonzero(as_tuple=False).flatten()
    valid_utility = utility[valid_indices]
    endpoint_cap = max(1, int(config.max_verify_sequences))
    endpoint_floor = min(endpoint_cap, max(1, int(config.min_verify_sequences)))
    candidate_take = min(
        int(valid_indices.numel()),
        max(endpoint_floor, endpoint_cap * int(config.selection_top_multiplier)),
    )
    order = valid_indices[torch.topk(valid_utility, k=candidate_take, largest=True).indices] + 1
    selected_with_root = _closure_mask_from_ranked_nodes(
        order,
        trie.parents,
        max_nodes,
        min_nodes,
        max_depth,
        max_endpoints=endpoint_cap,
    )

    if int(selected_with_root[1:].sum().item()) < min_nodes:
        marginal_take = min(int(trie.num_nodes), endpoint_cap * int(config.selection_top_multiplier))
        marginal_order = torch.argsort(trie.cum_log_probs, descending=True)[:marginal_take] + 1
        selected_with_root = _closure_mask_from_ranked_nodes(
            marginal_order,
            trie.parents,
            max_nodes,
            min_nodes,
            max_depth,
            max_endpoints=endpoint_cap,
        )
    return selected_with_root[1:]


@torch.inference_mode()
def score_candidate_trie(
    trie: CandidateTrie,
    lattice: TopKLattice,
    root_token_id: torch.Tensor,
    model: NodeValueNet,
    config: JointDDTConfig,
    prompt_length: int,
    context_hidden: torch.Tensor | None = None,
    calibration: dict | None = None,
) -> EdgeScoreResult:
    batch = build_edge_feature_batch(trie, lattice, root_token_id, model, context_hidden=context_hidden)
    edge_logits, other_logits = model(batch)
    edge_logits, other_logits = apply_logit_calibration(edge_logits, other_logits, trie.depths, calibration)
    if trie.num_nodes > 0:
        q_cond, q_other = grouped_softmax(edge_logits, trie.edge_parent_ids, other_logits)
        q_reach = propagate_reach(trie, q_cond, max_depth=lattice.horizon)
    else:
        q_cond = edge_logits.new_empty(0)
        q_other = other_logits.softmax(dim=0) if other_logits.numel() else edge_logits.new_empty(0)
        q_reach = torch.ones((1,), dtype=edge_logits.dtype, device=trie.device)
    return EdgeScoreResult(
        edge_logits=edge_logits,
        other_logits=other_logits,
        q_cond=q_cond,
        q_other=q_other,
        q_reach=q_reach,
    )


@torch.inference_mode()
def select_joint_tree(
    trie: CandidateTrie,
    lattice: TopKLattice,
    root_token_id: torch.Tensor,
    model: NodeValueNet,
    config: JointDDTConfig,
    prompt_length: int,
    context_hidden: torch.Tensor | None = None,
    calibration: dict | None = None,
) -> JointSelectionResult:
    scores = score_candidate_trie(
        trie,
        lattice,
        root_token_id,
        model,
        config,
        prompt_length,
        context_hidden=context_hidden,
        calibration=calibration,
    )
    edge_logits = scores.edge_logits
    other_logits = scores.other_logits
    q_cond = scores.q_cond
    q_other = scores.q_other
    q_reach = scores.q_reach

    best_reach = float(q_reach[1:].max().item()) if trie.num_nodes > 0 else 0.0
    fallback_reason = None
    if best_reach < config.min_best_reach_for_joint:
        fallback_reason = "low_best_reach"
    if calibration and float(calibration.get("confidence", 1.0)) < float(config.min_calibration_confidence):
        fallback_reason = fallback_reason or "low_calibration_confidence"

    selected_mask = _select_prefix_closed(trie, q_reach, config, prompt_length=prompt_length, max_depth=lattice.horizon)
    if int(selected_mask.sum().item()) < config.min_useful_verify_nodes:
        fallback_reason = fallback_reason or "below_min_useful_nodes"

    selected_tree = compact_selected_trie(trie, selected_mask, max_depth=lattice.horizon)
    metrics = {
        "candidate_nodes": trie.num_nodes,
        "selected_nodes": selected_tree.num_nodes,
        "selected_sequences": _leaf_count(selected_tree.parents),
        "best_reach": best_reach,
        "mean_selected_reach": float(q_reach[1:][selected_mask].mean().item()) if selected_mask.any() else 0.0,
        "fallback_reason": fallback_reason or "",
    }
    return JointSelectionResult(
        selected_tree=selected_tree,
        selected_mask=selected_mask,
        q_cond=q_cond,
        q_other=q_other,
        q_reach=q_reach,
        edge_logits=edge_logits,
        other_logits=other_logits,
        metrics=metrics,
        fallback_reason=fallback_reason,
    )


def select_marginal_tree(
    trie: CandidateTrie,
    config: JointDDTConfig,
    max_depth: int | None = None,
) -> SelectedTree:
    selected = torch.zeros((trie.num_nodes,), dtype=torch.bool, device=trie.device)
    if trie.num_nodes == 0 or config.max_verify_nodes <= 0:
        return compact_selected_trie(trie, selected, max_depth=max_depth)
    if max_depth is None:
        max_depth = int(trie.depths.max().item()) if trie.depths.numel() else 1
    endpoint_cap = max(1, min(int(config.max_verify_sequences), trie.num_nodes))
    order = torch.argsort(trie.cum_log_probs, descending=True)[:endpoint_cap] + 1
    selected_with_root = _closure_mask_from_ranked_nodes(
        order,
        trie.parents,
        max_nonroot_nodes=int(config.max_verify_nodes),
        min_nonroot_nodes=min(int(config.min_verify_nodes), int(config.max_verify_nodes), trie.num_nodes),
        max_depth=max_depth,
        max_endpoints=endpoint_cap,
    )
    selected = selected_with_root[1:]
    return compact_selected_trie(trie, selected, max_depth=max_depth)
