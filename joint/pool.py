from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import JointDDTConfig
from .lattice import TopKLattice


SOURCE_MARGINAL = 0
SOURCE_VALUE_BEAM = 1
SOURCE_DIVERSE = 2
SOURCE_ENTROPY = 3


@dataclass
class CandidateTrie:
    token_ids: torch.Tensor
    depths: torch.Tensor
    parents: torch.Tensor
    ranks: torch.Tensor
    step_log_probs: torch.Tensor
    cum_log_probs: torch.Tensor
    source_ids: torch.Tensor
    path_hashes: torch.Tensor | None = None

    @property
    def device(self) -> torch.device:
        return self.parents.device

    @property
    def num_nodes(self) -> int:
        return int(self.token_ids.numel())

    @property
    def num_total_nodes(self) -> int:
        return int(self.parents.numel())

    @property
    def edge_parent_ids(self) -> torch.Tensor:
        return self.parents[1:]

    def empty_like_selected(self) -> "CandidateTrie":
        device = self.device
        return CandidateTrie(
            token_ids=torch.empty(0, dtype=torch.long, device=device),
            depths=torch.empty(0, dtype=torch.long, device=device),
            parents=torch.tensor([-1], dtype=torch.long, device=device),
            ranks=torch.empty(0, dtype=torch.long, device=device),
            step_log_probs=torch.empty(0, dtype=torch.float32, device=device),
            cum_log_probs=torch.empty(0, dtype=torch.float32, device=device),
            source_ids=torch.empty(0, dtype=torch.long, device=device),
            path_hashes=torch.empty(0, dtype=torch.long, device=device),
        )


@torch.no_grad()
def build_marginal_candidate_trie(
    lattice: TopKLattice,
    config: JointDDTConfig,
    source_id: int = SOURCE_MARGINAL,
    max_nodes: int | None = None,
    width: int | None = None,
    rank_penalty: float = 0.0,
    entropy_bonus: float = 0.0,
    first_token_cap_fraction: float | None = None,
    seed_rank_count: int | None = None,
    score_log_probs: torch.Tensor | None = None,
) -> CandidateTrie:
    device = lattice.top_token_ids.device
    max_nodes = int(config.candidate_pool_nodes if max_nodes is None else max_nodes)
    if max_nodes <= 0 or lattice.horizon == 0:
        return CandidateTrie(
            token_ids=torch.empty(0, dtype=torch.long, device=device),
            depths=torch.empty(0, dtype=torch.long, device=device),
            parents=torch.tensor([-1], dtype=torch.long, device=device),
            ranks=torch.empty(0, dtype=torch.long, device=device),
            step_log_probs=torch.empty(0, dtype=torch.float32, device=device),
            cum_log_probs=torch.empty(0, dtype=torch.float32, device=device),
            source_ids=torch.empty(0, dtype=torch.long, device=device),
            path_hashes=torch.empty(0, dtype=torch.long, device=device),
        )

    width = max(1, int(config.candidate_pool_sequences if width is None else width))
    depth_limit = lattice.horizon if config.max_depth is None else min(lattice.horizon, int(config.max_depth))
    token_chunks: list[torch.Tensor] = []
    depth_chunks: list[torch.Tensor] = []
    parent_chunks: list[torch.Tensor] = [torch.tensor([-1], dtype=torch.long, device=device)]
    rank_chunks: list[torch.Tensor] = []
    step_log_prob_chunks: list[torch.Tensor] = []
    cum_log_prob_chunks: list[torch.Tensor] = []
    source_chunks: list[torch.Tensor] = []
    hash_chunks: list[torch.Tensor] = []

    prev_node_indices = torch.zeros(1, dtype=torch.long, device=device)
    prev_score = torch.zeros(1, dtype=torch.float32, device=device)
    prev_dflash_cum = torch.zeros(1, dtype=torch.float32, device=device)
    all_hashes = torch.zeros(1, dtype=torch.long, device=device)
    total_nodes = 0
    k = lattice.topk
    hash_base = torch.tensor(1000003, dtype=torch.long, device=device)

    for depth in range(1, depth_limit + 1):
        remaining = max_nodes - total_nodes
        if remaining <= 0:
            break

        parent_count = int(prev_node_indices.numel())
        rank_limit = k if depth > 1 or seed_rank_count is None else min(k, max(1, int(seed_rank_count)))
        parent_ids = prev_node_indices.repeat_interleave(rank_limit)
        parent_score = prev_score.repeat_interleave(rank_limit)
        parent_dflash_cum = prev_dflash_cum.repeat_interleave(rank_limit)
        ranks = torch.arange(rank_limit, dtype=torch.long, device=device).repeat(parent_count)
        step_log_probs = lattice.top_log_probs[depth - 1].gather(0, ranks).float()
        ranking_source = lattice.top_log_probs if score_log_probs is None else score_log_probs
        score_step_log_probs = ranking_source[depth - 1].gather(0, ranks).float()
        entropy_term = float(entropy_bonus) * lattice.position_entropy[depth - 1].float()
        scores = parent_score + score_step_log_probs - float(rank_penalty) * ranks.float() + entropy_term

        take = min(remaining, width, int(scores.numel()))
        if take <= 0:
            break
        selected_scores, selected = torch.topk(scores, k=take, dim=0)
        selected_parent_ids = parent_ids.gather(0, selected)
        selected_ranks = ranks.gather(0, selected)
        selected_step_log_probs = step_log_probs.gather(0, selected)
        selected_cum_log_probs = parent_dflash_cum.gather(0, selected) + selected_step_log_probs
        selected_token_ids = lattice.top_token_ids[depth - 1].gather(0, selected_ranks)
        selected_parent_hashes = all_hashes.gather(0, selected_parent_ids)
        selected_hashes = selected_parent_hashes * hash_base + selected_token_ids.long() + 17

        new_node_indices = torch.arange(
            total_nodes + 1,
            total_nodes + 1 + take,
            dtype=torch.long,
            device=device,
        )

        token_chunks.append(selected_token_ids)
        depth_chunks.append(torch.full((take,), depth, dtype=torch.long, device=device))
        parent_chunks.append(selected_parent_ids)
        rank_chunks.append(selected_ranks)
        step_log_prob_chunks.append(selected_step_log_probs)
        cum_log_prob_chunks.append(selected_cum_log_probs.float())
        source_chunks.append(torch.full((take,), source_id, dtype=torch.long, device=device))
        hash_chunks.append(selected_hashes.long())

        total_nodes += take
        prev_node_indices = new_node_indices
        prev_score = selected_scores.float()
        prev_dflash_cum = selected_cum_log_probs.float()
        all_hashes = torch.cat([all_hashes, selected_hashes.long()], dim=0)

    if total_nodes == 0:
        return CandidateTrie(
            token_ids=torch.empty(0, dtype=torch.long, device=device),
            depths=torch.empty(0, dtype=torch.long, device=device),
            parents=torch.tensor([-1], dtype=torch.long, device=device),
            ranks=torch.empty(0, dtype=torch.long, device=device),
            step_log_probs=torch.empty(0, dtype=torch.float32, device=device),
            cum_log_probs=torch.empty(0, dtype=torch.float32, device=device),
            source_ids=torch.empty(0, dtype=torch.long, device=device),
            path_hashes=torch.empty(0, dtype=torch.long, device=device),
        )

    trie = CandidateTrie(
        token_ids=torch.cat(token_chunks, dim=0).long(),
        depths=torch.cat(depth_chunks, dim=0).long(),
        parents=torch.cat(parent_chunks, dim=0).long(),
        ranks=torch.cat(rank_chunks, dim=0).long(),
        step_log_probs=torch.cat(step_log_prob_chunks, dim=0).float(),
        cum_log_probs=torch.cat(cum_log_prob_chunks, dim=0).float(),
        source_ids=torch.cat(source_chunks, dim=0).long(),
        path_hashes=torch.cat(hash_chunks, dim=0).long(),
    )
    if first_token_cap_fraction is not None and trie.num_nodes > 0:
        return cap_first_token_modes(trie, max_nodes=max_nodes, cap_fraction=float(first_token_cap_fraction))
    return trie


def _empty_trie(device: torch.device) -> CandidateTrie:
    return CandidateTrie(
        token_ids=torch.empty(0, dtype=torch.long, device=device),
        depths=torch.empty(0, dtype=torch.long, device=device),
        parents=torch.tensor([-1], dtype=torch.long, device=device),
        ranks=torch.empty(0, dtype=torch.long, device=device),
        step_log_probs=torch.empty(0, dtype=torch.float32, device=device),
        cum_log_probs=torch.empty(0, dtype=torch.float32, device=device),
        source_ids=torch.empty(0, dtype=torch.long, device=device),
        path_hashes=torch.empty(0, dtype=torch.long, device=device),
    )


def ddtree_tree_to_candidate_trie(
    *,
    draft_logits: torch.Tensor,
    node_token_ids: torch.Tensor,
    node_depths: torch.Tensor,
    parents: list[int],
    tree_budget: int,
    source_id: int = SOURCE_MARGINAL,
    top_log_probs: torch.Tensor | None = None,
    top_token_ids_precomputed: torch.Tensor | None = None,
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

    if top_log_probs is None or top_token_ids_precomputed is None:
        rank_k = min(max(int(tree_budget), num_nodes, 1), int(draft_logits.shape[-1]))
        logits = draft_logits.float()
        top_logits, top_token_ids_precomputed = torch.topk(logits, k=rank_k, dim=-1)
        top_log_probs = top_logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    else:
        top_log_probs = top_log_probs.to(device=device)
        top_token_ids_precomputed = top_token_ids_precomputed.to(device=device)

    depth_indices = depths - 1
    per_node_top_tokens = top_token_ids_precomputed.index_select(0, depth_indices)
    matches = per_node_top_tokens == token_ids.unsqueeze(1)
    if not bool(matches.any(dim=1).all().item()):
        missing = (~matches.any(dim=1)).nonzero(as_tuple=False).flatten()[:8].detach().cpu().tolist()
        raise ValueError(
            "DDTree candidate conversion failed: some nodes were not found in the same-budget top-k list; "
            f"tree_budget={tree_budget}, rank_k={top_log_probs.shape[-1]}, missing_edges={missing}"
        )
    ranks = matches.float().argmax(dim=1).long()
    step_log_probs = top_log_probs[depth_indices, ranks].float()

    # Vectorized cum_log_probs: process depth-by-depth on GPU (no Python-per-node loop)
    max_depth = int(depths.max().item()) if depths.numel() else 0
    cum_all = torch.zeros(num_nodes + 1, dtype=torch.float32, device=device)
    for d in range(1, max_depth + 1):
        edge_mask = depths == d
        if not edge_mask.any():
            continue
        edges = edge_mask.nonzero(as_tuple=False).flatten()
        node_ids = edges + 1
        parent_ids = parent_tensor[node_ids]
        parent_cum = torch.where(parent_ids <= 0,
                                 torch.zeros_like(parent_ids, dtype=torch.float32),
                                 cum_all[parent_ids])
        cum_all[node_ids] = parent_cum + step_log_probs[edges]
    cum_log_probs = cum_all[1:]

    return CandidateTrie(
        token_ids=token_ids,
        depths=depths,
        parents=parent_tensor,
        ranks=ranks,
        step_log_probs=step_log_probs,
        cum_log_probs=cum_log_probs,
        source_ids=torch.full((num_nodes,), int(source_id), dtype=torch.long, device=device),
        path_hashes=torch.arange(1, num_nodes + 1, dtype=torch.long, device=device),
    )


@torch.no_grad()
def build_ddtree_candidate_trie(draft_logits: torch.Tensor, tree_budget: int) -> CandidateTrie:
    import heapq
    import numpy as np

    budget = int(tree_budget)
    if budget <= 0 or draft_logits.shape[0] == 0:
        device = draft_logits.device
        return CandidateTrie(
            token_ids=torch.empty(0, dtype=torch.long, device=device),
            depths=torch.empty(0, dtype=torch.long, device=device),
            parents=torch.tensor([-1], dtype=torch.long, device=device),
            ranks=torch.empty(0, dtype=torch.long, device=device),
            step_log_probs=torch.empty(0, dtype=torch.float32, device=device),
            cum_log_probs=torch.empty(0, dtype=torch.float32, device=device),
            source_ids=torch.empty(0, dtype=torch.long, device=device),
            path_hashes=torch.empty(0, dtype=torch.long, device=device),
        )

    device = draft_logits.device
    topk = min(budget, draft_logits.shape[-1])
    depth_limit = int(draft_logits.shape[0])

    # Compute topk on GPU, single bulk transfer to CPU
    logits = draft_logits.float()
    top_logits, top_token_ids_gpu = torch.topk(logits, k=topk, dim=-1)
    top_log_probs_gpu = top_logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    top_log_probs_np = top_log_probs_gpu.cpu().numpy()
    top_token_ids_np = top_token_ids_gpu.cpu().numpy()

    # Heap loop: compute ALL trie fields on CPU (no post-processing needed)
    node_token_ids_np = np.empty(budget, dtype=np.int64)
    node_depths_np = np.empty(budget, dtype=np.int64)
    node_ranks_np = np.empty(budget, dtype=np.int64)
    node_step_lp_np = np.empty(budget, dtype=np.float32)
    node_cum_lp_np = np.empty(budget, dtype=np.float32)
    cum_lp_by_index = np.zeros(budget + 1, dtype=np.float32)
    parents_list = [-1]
    node_count = 0

    first_logw = float(top_log_probs_np[0, 0])
    heap: list[tuple[float, tuple[int, ...], int, int, int, float]] = [(-first_logw, (0,), 0, 1, 0, first_logw)]

    while heap and node_count < budget:
        _, ranks_tuple, parent_index, depth, rank, logw = heapq.heappop(heap)
        step_lp = float(top_log_probs_np[depth - 1, rank])
        cum_lp = cum_lp_by_index[parent_index] + step_lp

        node_token_ids_np[node_count] = int(top_token_ids_np[depth - 1, rank])
        node_depths_np[node_count] = depth
        node_ranks_np[node_count] = rank
        node_step_lp_np[node_count] = step_lp
        node_cum_lp_np[node_count] = cum_lp

        parents_list.append(parent_index)
        current_index = node_count + 1
        cum_lp_by_index[current_index] = cum_lp
        node_count += 1

        if rank + 1 < topk:
            sibling_logw = logw - step_lp + float(top_log_probs_np[depth - 1, rank + 1])
            heapq.heappush(heap, (-sibling_logw, ranks_tuple[:-1] + (rank + 1,), parent_index, depth, rank + 1, sibling_logw))
        if depth < depth_limit:
            child_logw = logw + float(top_log_probs_np[depth, 0])
            heapq.heappush(heap, (-child_logw, ranks_tuple + (0,), current_index, depth + 1, 0, child_logw))

    # Single bulk transfer to GPU (no ddtree_tree_to_candidate_trie needed)
    n = node_count
    return CandidateTrie(
        token_ids=torch.from_numpy(node_token_ids_np[:n]).to(device=device, dtype=torch.long),
        depths=torch.from_numpy(node_depths_np[:n]).to(device=device, dtype=torch.long),
        parents=torch.tensor(parents_list, dtype=torch.long, device=device),
        ranks=torch.from_numpy(node_ranks_np[:n]).to(device=device, dtype=torch.long),
        step_log_probs=torch.from_numpy(node_step_lp_np[:n]).to(device=device, dtype=torch.float32),
        cum_log_probs=torch.from_numpy(node_cum_lp_np[:n]).to(device=device, dtype=torch.float32),
        source_ids=torch.full((n,), SOURCE_MARGINAL, dtype=torch.long, device=device),
        path_hashes=torch.arange(1, n + 1, dtype=torch.long, device=device),
    )


def cap_first_token_modes(trie: CandidateTrie, max_nodes: int, cap_fraction: float) -> CandidateTrie:
    if trie.num_nodes == 0:
        return trie
    device = trie.device
    all_first_rank = torch.full((trie.num_total_nodes,), -1, dtype=torch.long, device=device)
    depth1_nodes = (trie.depths == 1).nonzero(as_tuple=False).flatten() + 1
    all_first_rank[depth1_nodes] = trie.ranks[depth1_nodes - 1]
    max_depth = int(trie.depths.max().item())
    for depth in range(2, max_depth + 1):
        nodes = (trie.depths == depth).nonzero(as_tuple=False).flatten() + 1
        if nodes.numel():
            all_first_rank[nodes] = all_first_rank[trie.parents[nodes]]
    first_rank = all_first_rank[1:]
    cap = max(1, int(max_nodes * cap_fraction))
    keep = torch.zeros((trie.num_nodes,), dtype=torch.bool, device=device)
    for rank in torch.unique(first_rank):
        if int(rank.item()) < 0:
            continue
        indices = (first_rank == rank).nonzero(as_tuple=False).flatten()
        scores = trie.cum_log_probs[indices]
        take = min(cap, int(indices.numel()))
        chosen = indices[torch.topk(scores, k=take).indices]
        keep[chosen] = True
    return compact_candidate_trie_by_mask(trie, keep)


def compact_candidate_trie_by_mask(trie: CandidateTrie, keep_mask: torch.Tensor) -> CandidateTrie:
    keep_mask = keep_mask.to(device=trie.device, dtype=torch.bool)
    keep_with_root = torch.cat([torch.ones(1, dtype=torch.bool, device=trie.device), keep_mask], dim=0)
    max_depth = int(trie.depths.max().item()) if trie.depths.numel() else 1
    for _ in range(max_depth):
        parent_keep = keep_with_root[trie.parents.clamp_min(0)]
        keep_with_root = keep_with_root & parent_keep
        keep_with_root[0] = True
    old_ids = keep_with_root.nonzero(as_tuple=False).flatten()
    if old_ids.numel() == 1:
        return _empty_trie(trie.device)
    old_to_new = torch.full((trie.num_total_nodes,), -1, dtype=torch.long, device=trie.device)
    old_to_new[old_ids] = torch.arange(old_ids.numel(), dtype=torch.long, device=trie.device)
    nonroot_old = old_ids[1:]
    edge_old = nonroot_old - 1
    path_hashes = None if trie.path_hashes is None else trie.path_hashes[edge_old]
    return CandidateTrie(
        token_ids=trie.token_ids[edge_old],
        depths=trie.depths[edge_old],
        parents=torch.cat([torch.tensor([-1], dtype=torch.long, device=trie.device), old_to_new[trie.parents[nonroot_old]]], dim=0),
        ranks=trie.ranks[edge_old],
        step_log_probs=trie.step_log_probs[edge_old],
        cum_log_probs=trie.cum_log_probs[edge_old],
        source_ids=trie.source_ids[edge_old],
        path_hashes=path_hashes,
    )


def merge_candidate_tries(tries: list[CandidateTrie], max_nodes: int) -> CandidateTrie:
    tries = [trie for trie in tries if trie.num_nodes > 0]
    if not tries:
        raise ValueError("merge_candidate_tries requires at least one non-empty trie")
    device = tries[0].device
    token_chunks: list[torch.Tensor] = []
    depth_chunks: list[torch.Tensor] = []
    parent_chunks: list[torch.Tensor] = [torch.tensor([-1], dtype=torch.long, device=device)]
    rank_chunks: list[torch.Tensor] = []
    step_chunks: list[torch.Tensor] = []
    cum_chunks: list[torch.Tensor] = []
    source_chunks: list[torch.Tensor] = []
    hash_chunks: list[torch.Tensor] = []
    offset = 0
    for trie in tries:
        remaining = max_nodes - offset
        if remaining <= 0:
            break
        take = min(remaining, trie.num_nodes)
        edge_slice = slice(0, take)
        parents = trie.parents[1 : take + 1]
        shifted_parents = torch.where(parents == 0, parents, parents + offset)
        token_chunks.append(trie.token_ids[edge_slice])
        depth_chunks.append(trie.depths[edge_slice])
        parent_chunks.append(shifted_parents)
        rank_chunks.append(trie.ranks[edge_slice])
        step_chunks.append(trie.step_log_probs[edge_slice])
        cum_chunks.append(trie.cum_log_probs[edge_slice])
        source_chunks.append(trie.source_ids[edge_slice])
        if trie.path_hashes is None:
            hash_chunks.append(torch.arange(offset + 1, offset + take + 1, dtype=torch.long, device=device))
        else:
            hash_chunks.append(trie.path_hashes[edge_slice])
        offset += take

    if offset == 0:
        return _empty_trie(device)
    merged = CandidateTrie(
        token_ids=torch.cat(token_chunks, dim=0),
        depths=torch.cat(depth_chunks, dim=0),
        parents=torch.cat(parent_chunks, dim=0),
        ranks=torch.cat(rank_chunks, dim=0),
        step_log_probs=torch.cat(step_chunks, dim=0),
        cum_log_probs=torch.cat(cum_chunks, dim=0),
        source_ids=torch.cat(source_chunks, dim=0),
        path_hashes=torch.cat(hash_chunks, dim=0),
    )
    return deduplicate_candidate_trie(merged)


def deduplicate_candidate_trie(trie: CandidateTrie) -> CandidateTrie:
    if trie.num_nodes <= 1:
        return trie
    parents_cpu = [int(x) for x in trie.parents.detach().cpu().tolist()]
    tokens_cpu = [int(x) for x in trie.token_ids.detach().cpu().tolist()]
    source_cpu = [int(x) for x in trie.source_ids.detach().cpu().tolist()]
    old_to_new: dict[int, int] = {0: 0}
    edge_to_new: dict[tuple[int, int], int] = {}
    kept_old_nodes: list[int] = []
    kept_source_ids: list[int] = []
    for old_node in range(1, trie.num_total_nodes):
        parent_old = parents_cpu[old_node]
        parent_new = old_to_new.get(parent_old)
        if parent_new is None:
            continue
        token = tokens_cpu[old_node - 1]
        key = (parent_new, token)
        existing = edge_to_new.get(key)
        source_id = source_cpu[old_node - 1]
        if existing is not None:
            old_to_new[old_node] = existing
            if source_id != SOURCE_MARGINAL:
                kept_source_ids[existing - 1] = source_id
            continue
        new_node = len(kept_old_nodes) + 1
        old_to_new[old_node] = new_node
        edge_to_new[key] = new_node
        kept_old_nodes.append(old_node)
        kept_source_ids.append(source_id)

    if len(kept_old_nodes) == trie.num_nodes:
        return trie
    if not kept_old_nodes:
        return _empty_trie(trie.device)

    edge_old = torch.tensor([old_node - 1 for old_node in kept_old_nodes], dtype=torch.long, device=trie.device)
    new_parents = torch.tensor(
        [-1] + [old_to_new[parents_cpu[old_node]] for old_node in kept_old_nodes],
        dtype=torch.long,
        device=trie.device,
    )
    path_hashes = None if trie.path_hashes is None else trie.path_hashes[edge_old]
    return CandidateTrie(
        token_ids=trie.token_ids[edge_old],
        depths=trie.depths[edge_old],
        parents=new_parents,
        ranks=trie.ranks[edge_old],
        step_log_probs=trie.step_log_probs[edge_old],
        cum_log_probs=trie.cum_log_probs[edge_old],
        source_ids=torch.tensor(kept_source_ids, dtype=torch.long, device=trie.device),
        path_hashes=path_hashes,
    )


@torch.no_grad()
def build_union_candidate_trie(
    lattice: TopKLattice,
    config: JointDDTConfig,
    value_token_scores: torch.Tensor | None = None,
) -> CandidateTrie:
    max_nodes = int(config.candidate_pool_nodes)
    parts: list[CandidateTrie] = []
    marginal_nodes = max(1, int(max_nodes * config.marginal_pool_fraction))
    parts.append(build_marginal_candidate_trie(lattice, config, SOURCE_MARGINAL, max_nodes=marginal_nodes))

    if config.enable_value_beam_pool:
        value_nodes = max(1, int(max_nodes * config.value_pool_fraction))
        score_log_probs = None
        if value_token_scores is not None:
            score_log_probs = lattice.top_log_probs + float(config.value_beam_logit_weight) * value_token_scores
        parts.append(
            build_marginal_candidate_trie(
                lattice,
                config,
                SOURCE_VALUE_BEAM,
                max_nodes=value_nodes,
                rank_penalty=0.01,
                score_log_probs=score_log_probs,
            )
        )

    if config.enable_diversity_pool:
        diversity_nodes = max(1, int(max_nodes * config.diversity_pool_fraction))
        parts.append(
            build_marginal_candidate_trie(
                lattice,
                config,
                SOURCE_DIVERSE,
                max_nodes=diversity_nodes,
                width=max(config.joint_topk, min(config.candidate_pool_sequences, diversity_nodes)),
                rank_penalty=-0.005,
                first_token_cap_fraction=config.diversity_first_token_cap_fraction,
                seed_rank_count=min(lattice.topk, max(config.joint_topk, 1)),
            )
        )

    if config.enable_entropy_pool:
        entropy_nodes = max(1, int(max_nodes * config.entropy_pool_fraction))
        parts.append(
            build_marginal_candidate_trie(
                lattice,
                config,
                SOURCE_ENTROPY,
                max_nodes=entropy_nodes,
                width=max(config.joint_topk, min(config.candidate_pool_sequences, entropy_nodes)),
                rank_penalty=-0.01,
                entropy_bonus=config.entropy_bonus,
            )
        )

    return merge_candidate_tries(parts, max_nodes=max_nodes)


def value_token_scores_from_edges(
    lattice: TopKLattice,
    trie: CandidateTrie,
    edge_logits: torch.Tensor,
) -> torch.Tensor:
    scores = torch.zeros_like(lattice.top_log_probs, dtype=torch.float32)
    if trie.num_nodes == 0:
        return scores
    flat_size = lattice.horizon * lattice.topk
    flat_scores = torch.full((flat_size,), -torch.inf, dtype=torch.float32, device=trie.device)
    flat_index = (trie.depths - 1).clamp(0, lattice.horizon - 1) * lattice.topk + trie.ranks.clamp(0, lattice.topk - 1)
    if hasattr(flat_scores, "scatter_reduce_"):
        flat_scores.scatter_reduce_(0, flat_index, edge_logits.float(), reduce="amax", include_self=True)
    else:
        for idx in range(int(flat_index.numel())):
            flat_scores[flat_index[idx]] = torch.maximum(flat_scores[flat_index[idx]], edge_logits[idx].float())
    flat_scores = flat_scores.view(lattice.horizon, lattice.topk)
    flat_scores = torch.where(torch.isfinite(flat_scores), flat_scores, torch.zeros_like(flat_scores))
    flat_scores = flat_scores - flat_scores.mean(dim=-1, keepdim=True)
    return flat_scores.clamp(min=-8.0, max=8.0)


def candidate_trie_to_cpu_dict(trie: CandidateTrie) -> dict[str, list[int] | list[float]]:
    return {
        "token_ids": [int(x) for x in trie.token_ids.detach().cpu().tolist()],
        "depths": [int(x) for x in trie.depths.detach().cpu().tolist()],
        "parents": [int(x) for x in trie.parents.detach().cpu().tolist()],
        "ranks": [int(x) for x in trie.ranks.detach().cpu().tolist()],
        "step_log_probs": [float(x) for x in trie.step_log_probs.detach().cpu().tolist()],
        "cum_log_probs": [float(x) for x in trie.cum_log_probs.detach().cpu().tolist()],
        "source_ids": [int(x) for x in trie.source_ids.detach().cpu().tolist()],
        "path_hashes": [] if trie.path_hashes is None else [int(x) for x in trie.path_hashes.detach().cpu().tolist()],
    }


def candidate_trie_from_dict(data: dict, device: torch.device | str = "cpu") -> CandidateTrie:
    return CandidateTrie(
        token_ids=torch.tensor(data.get("token_ids", []), dtype=torch.long, device=device),
        depths=torch.tensor(data.get("depths", []), dtype=torch.long, device=device),
        parents=torch.tensor(data.get("parents", [-1]), dtype=torch.long, device=device),
        ranks=torch.tensor(data.get("ranks", []), dtype=torch.long, device=device),
        step_log_probs=torch.tensor(data.get("step_log_probs", []), dtype=torch.float32, device=device),
        cum_log_probs=torch.tensor(data.get("cum_log_probs", []), dtype=torch.float32, device=device),
        source_ids=torch.tensor(data.get("source_ids", []), dtype=torch.long, device=device),
        path_hashes=torch.tensor(data.get("path_hashes", []), dtype=torch.long, device=device)
        if data.get("path_hashes") else None,
    )
