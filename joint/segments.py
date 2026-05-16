from __future__ import annotations

import torch
import torch.nn.functional as F


def segment_amax(values: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    out = torch.full((num_segments,), -torch.inf, dtype=values.dtype, device=values.device)
    if hasattr(out, "scatter_reduce_"):
        return out.scatter_reduce_(0, segment_ids, values, reduce="amax", include_self=True)
    for segment in range(num_segments):
        mask = segment_ids == segment
        if mask.any():
            out[segment] = values[mask].max()
    return out


def segment_sum(values: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    out = torch.zeros((num_segments,), dtype=values.dtype, device=values.device)
    return out.scatter_add_(0, segment_ids, values)


def grouped_log_softmax(
    edge_logits: torch.Tensor,
    parent_ids: torch.Tensor,
    other_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if edge_logits.dim() != 1:
        raise ValueError("edge_logits must be 1D")
    if parent_ids.shape != edge_logits.shape:
        raise ValueError("parent_ids must have the same shape as edge_logits")
    if other_logits.dim() != 1:
        raise ValueError("other_logits must be 1D")

    num_parents = int(other_logits.numel())
    other_parent_ids = torch.arange(num_parents, dtype=parent_ids.dtype, device=parent_ids.device)
    all_logits = torch.cat([edge_logits, other_logits], dim=0)
    all_parent_ids = torch.cat([parent_ids, other_parent_ids], dim=0)

    max_by_parent = segment_amax(all_logits, all_parent_ids, num_parents)
    shifted = all_logits - max_by_parent[all_parent_ids]
    exp_shifted = shifted.exp()
    denom = segment_sum(exp_shifted, all_parent_ids, num_parents).clamp_min(torch.finfo(exp_shifted.dtype).tiny)
    all_log_probs = shifted - denom[all_parent_ids].log()

    edge_count = edge_logits.numel()
    return all_log_probs[:edge_count], all_log_probs[edge_count:]


def grouped_softmax(
    edge_logits: torch.Tensor,
    parent_ids: torch.Tensor,
    other_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    edge_log_probs, other_log_probs = grouped_log_softmax(edge_logits, parent_ids, other_logits)
    return edge_log_probs.exp(), other_log_probs.exp()


def grouped_cross_entropy(
    edge_logits: torch.Tensor,
    parent_ids: torch.Tensor,
    other_logits: torch.Tensor,
    target_edge_indices: torch.Tensor,
) -> torch.Tensor:
    edge_log_probs, other_log_probs = grouped_log_softmax(edge_logits, parent_ids, other_logits)
    target_edge_indices = target_edge_indices.to(device=edge_logits.device, dtype=torch.long)
    is_other = target_edge_indices < 0
    chosen = other_log_probs.clone()
    if (~is_other).any():
        chosen[~is_other] = edge_log_probs[target_edge_indices[~is_other]]
    return -chosen.mean()


def grouped_kl_divergence(
    edge_logits: torch.Tensor,
    parent_ids: torch.Tensor,
    other_logits: torch.Tensor,
    target_edge_probs: torch.Tensor,
    target_other_probs: torch.Tensor,
) -> torch.Tensor:
    edge_log_probs, other_log_probs = grouped_log_softmax(edge_logits, parent_ids, other_logits)
    target_edge_probs = target_edge_probs.to(device=edge_logits.device, dtype=edge_logits.dtype).clamp_min(0)
    target_other_probs = target_other_probs.to(device=edge_logits.device, dtype=edge_logits.dtype).clamp_min(0)

    edge_kl = target_edge_probs * (target_edge_probs.clamp_min(1e-12).log() - edge_log_probs)
    other_kl = target_other_probs * (target_other_probs.clamp_min(1e-12).log() - other_log_probs)
    per_parent = segment_sum(edge_kl, parent_ids, other_logits.numel()) + other_kl
    return per_parent.mean()


def propagate_reach_from_edges(
    edge_probs: torch.Tensor,
    parent_ids: torch.Tensor,
    child_node_ids: torch.Tensor,
    root_node_ids: torch.Tensor,
    num_nodes: int,
    depths: torch.Tensor | None = None,
) -> torch.Tensor:
    if edge_probs.dim() != 1:
        raise ValueError("edge_probs must be 1D")
    if parent_ids.shape != edge_probs.shape:
        raise ValueError("parent_ids must have the same shape as edge_probs")
    if child_node_ids.shape != edge_probs.shape:
        raise ValueError("child_node_ids must have the same shape as edge_probs")

    q_reach = torch.zeros((int(num_nodes),), dtype=edge_probs.dtype, device=edge_probs.device)
    if root_node_ids.numel():
        q_reach[root_node_ids.to(device=edge_probs.device, dtype=torch.long)] = 1.0
    if edge_probs.numel() == 0:
        return q_reach

    parent_ids = parent_ids.to(device=edge_probs.device, dtype=torch.long)
    child_node_ids = child_node_ids.to(device=edge_probs.device, dtype=torch.long)
    if depths is not None and depths.numel():
        depths = depths.to(device=edge_probs.device, dtype=torch.long)
        for depth in range(1, int(depths.max().item()) + 1):
            mask = depths == depth
            if mask.any():
                q_reach[child_node_ids[mask]] = q_reach[parent_ids[mask]] * edge_probs[mask]
        return torch.minimum(q_reach, torch.ones_like(q_reach))

    for edge_index in torch.argsort(child_node_ids).detach().cpu().tolist():
        parent = parent_ids[edge_index]
        child = child_node_ids[edge_index]
        q_reach[child] = q_reach[parent] * edge_probs[edge_index]
    return torch.minimum(q_reach, torch.ones_like(q_reach))


def sibling_rank_loss(
    scores: torch.Tensor,
    parent_ids: torch.Tensor,
    positive_mask: torch.Tensor,
    margin: float = 0.05,
) -> torch.Tensor:
    if scores.numel() == 0:
        return scores.new_zeros(())
    if parent_ids.shape != scores.shape:
        raise ValueError("parent_ids must have the same shape as scores")
    if positive_mask.shape != scores.shape:
        raise ValueError("positive_mask must have the same shape as scores")

    positive_mask = positive_mask.to(device=scores.device, dtype=torch.bool)
    if not positive_mask.any():
        return scores.new_zeros(())

    losses = []
    parent_ids = parent_ids.to(device=scores.device, dtype=torch.long)
    for parent in torch.unique(parent_ids[positive_mask]):
        same_parent = parent_ids == parent
        positives = scores[same_parent & positive_mask]
        negatives = scores[same_parent & ~positive_mask]
        if positives.numel() == 0 or negatives.numel() == 0:
            continue
        best_positive = positives.max()
        losses.append(F.relu(float(margin) - best_positive + negatives).mean())
    if not losses:
        return scores.new_zeros(())
    return torch.stack(losses).mean()


def pairwise_rank_loss(positive_scores: torch.Tensor, negative_scores: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    if positive_scores.numel() == 0 or negative_scores.numel() == 0:
        return positive_scores.new_zeros(())
    count = min(positive_scores.numel(), negative_scores.numel())
    return F.relu(margin - positive_scores[:count] + negative_scores[:count]).mean()
